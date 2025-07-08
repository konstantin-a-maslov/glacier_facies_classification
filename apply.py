import tensorflow as tf
import cnn_model
import rioxarray
import xarray
import rasterio
import rasterio.features
import geopandas
import pickle
import scipy.ndimage
import numpy as np
import argparse
    

def apply(
    model_name, 
    raster_path,
    output_path,
    outlines_path,
    dem_path, 
    slope_path, 
    shadow_path,
    hillshade_path,
    proximity_path,
    confidence_path,
    calibration_model,
    no_smoothing, 
):
    if (dem_path or slope_path) and not (dem_path or slope_path):
        raise ValueError("Both elevation and slope rasters have to be provided.")
    if (shadow_path or hillshade_path) and not (shadow_path or hillshade_path):
        raise ValueError("Both shadow mask and hillshade rasters have to be provided.")
    if proximity_path and not (shadow_path or dem_path):
        raise ValueError("If proximity is provided, then elevation, slope, shadow mask and hillshade must be provided.")
    if shadow_path and not dem_path:
        raise ValueError("If shadow mask is provided, then elevation and slope must be provided.")
    
    FEATURE_INDICES = [0, 1, 2, 3, 4, 5] + \
        ([6] if dem_path else []) + \
        ([7] if slope_path else []) + \
        ([8] if shadow_path else []) + \
        ([9] if hillshade_path else []) + \
        ([10] if proximity_path else [])

    features = []
    raster = rioxarray.open_rasterio(raster_path)
    features.append(raster)
    
    outlines = geopandas.read_file(outlines_path)
    outlines = outlines.to_crs(raster.rio.crs)
    outlines.geometry = outlines.geometry.buffer(0) # curate just in case

    outlines_arr = rasterio.features.rasterize(
        ((geometry, 1) for geometry in outlines.geometry),
        out_shape=raster.data[0, ...].shape,
        transform=raster.rio.transform()
    )
    valid_coords = np.argwhere(outlines_arr)

    if dem_path:
        elevation = rioxarray.open_rasterio(dem_path)
        elevation_min = elevation.data[0, outlines_arr==1].min()
        elevation_max = elevation.data[0, outlines_arr==1].max()
        elevation = (elevation - elevation_min) / (elevation_max - elevation_min)
        features.append(elevation)
    if slope_path:
        slope = rioxarray.open_rasterio(slope_path)
        features.append(slope)
    if shadow_path:
        shadow = rioxarray.open_rasterio(shadow_path)
        features.append(shadow)
    if hillshade_path:
        hillshade = rioxarray.open_rasterio(hillshade_path)
        features.append(hillshade)
    if proximity_path:
        proximity = rioxarray.open_rasterio(proximity_path)
        proximity_max = proximity.data.max()
        proximity = proximity / proximity_max
        features.append(proximity)
    
    feature_arr = np.concatenate([_.data for _ in features], axis=0)
    feature_arr = np.moveaxis(feature_arr, 0, -1)
    with open("globalminmax.pickle", "rb") as minmax_src:
        mins, maxs = pickle.load(minmax_src)
    mins, maxs = mins[FEATURE_INDICES], maxs[FEATURE_INDICES]
    feature_arr = (feature_arr - mins) / (maxs - mins)
    
    inference_data = create_inference_dataset(feature_arr, valid_coords)

    model = cnn_model.build_cnn_model(input_features=len(FEATURE_INDICES), name=model_name)
    model.load_weights(f"weights/{model_name}_weights.h5")
    preds = model.predict(inference_data)
    if confidence_path:
        conf = np.max(preds, axis=-1)
        if calibration_model:
            with open(calibration_model, "rb") as src:
                calibration_model = pickle.load(src)
            conf = conf.reshape(-1, 1)
            conf = calibration_model.predict(conf)
    preds = np.argmax(preds, axis=-1)

    height, width, _ = feature_arr.shape
    out_arr = np.full((height, width), -9999, dtype=np.int16)
    conf_arr = np.full((height, width), 1.0, dtype=np.float32)
    for i, (y, x) in enumerate(valid_coords):
        out_arr[y, x] = preds[i]
        conf_arr[y, x] = conf[i]

    if (no_smoothing is False) or (no_smoothing is None):
        out_arr = smooth_predictions(out_arr, ignore_value=-9999)

    output = xarray.DataArray(
        data=out_arr,
        coords={"y": raster.coords["y"], "x": raster.coords["x"]},
        dims=["y", "x"],
    )
    output.rio.write_crs(raster.rio.crs, inplace=True)
    output.rio.write_transform(raster.rio.transform(), inplace=True)
    output.rio.set_nodata(-9999, inplace=True)
    output.rio.to_raster(output_path)
    if confidence_path:
        output_conf = xarray.DataArray(
            data=conf_arr,
            coords={"y": raster.coords["y"], "x": raster.coords["x"]},
            dims=["y", "x"],
        )
        output_conf.rio.write_crs(raster.rio.crs, inplace=True)
        output_conf.rio.write_transform(raster.rio.transform(), inplace=True)
        output_conf.rio.set_nodata(0, inplace=True)
        output_conf.rio.to_raster(confidence_path)
    

def patch_generator(feature_arr, valid_coords, patch_size=33):
    half_size = patch_size // 2
    height, width, _ = feature_arr.shape
    
    for (y, x) in valid_coords:
        # if (y - half_size < 0 or y + half_size >= height or
        #     x - half_size < 0 or x + half_size >= width):
        #     continue
        y_start, y_stop = y - half_size, y + half_size + 1
        x_start, x_stop = x - half_size, x + half_size + 1
        patch = feature_arr[y_start:y_stop, x_start:x_stop, ...]
        yield patch


def create_inference_dataset(feature_arr, valid_coords, patch_size=33, batch_size=1024):
    def gen():
        for patch in patch_generator(feature_arr, valid_coords, patch_size):
            yield patch
    
    dataset = tf.data.Dataset.from_generator(
        gen,
        output_types=tf.float32,
        output_shapes=(patch_size, patch_size, feature_arr.shape[-1]),
    )
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def smooth_predictions(preds, ignore_value=-9999):
    def mode_ignore_fill(values, ignore_value=ignore_value):
        valid_values = values[values != ignore_value]
        if len(valid_values) == 0:
            return ignore_value
        unique, counts = np.unique(valid_values, return_counts=True)
        return unique[np.argmax(counts)]

    custom_filter = lambda values: mode_ignore_fill(values, ignore_value)
    no_data_mask = (preds == ignore_value)
    filtered = scipy.ndimage.generic_filter(
        preds, function=custom_filter, 
        size=3, mode="constant", 
        cval=ignore_value
    )
    filtered[no_data_mask] = ignore_value
    return filtered


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", help="Model name")
    parser.add_argument("raster_path", help="Raster path (.tif)")
    parser.add_argument("output_path", help="Output path (.tif)")
    parser.add_argument("outlines_path", help="Outlines path (.shp/.geojson)")
    parser.add_argument("--dem_path", help="Elevation path (.tif)")
    parser.add_argument("--slope_path", help="Slope path (.tif)")
    parser.add_argument("--shadow_path", help="Shadow mask (0/1) path (.tif)")
    parser.add_argument("--hillshade_path", help="Hillshade path (.tif)")
    parser.add_argument("--proximity_path", help="Proximity path (.tif)")
    parser.add_argument("--confidence_path", help="Output confidence path (.tif)")
    parser.add_argument("--calibration_model", help="Confidence calibration model (.pickle)")
    parser.add_argument("--no_smoothing", action="store_true", help="Turn off mode smoothing of results")
    args = parser.parse_args()
    
    apply(
        args.model_name, 
        args.raster_path,
        args.output_path,
        args.outlines_path,
        args.dem_path, 
        args.slope_path, 
        args.shadow_path,
        args.hillshade_path,
        args.proximity_path,
        args.confidence_path,
        args.calibration_model,
        args.no_smoothing,
    )


if __name__ == "__main__":
    main()
