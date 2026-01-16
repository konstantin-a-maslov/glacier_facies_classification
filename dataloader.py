import tensorflow as tf
import rioxarray
import numpy as np
import random
import pickle
import glob
import os
from tqdm import tqdm


cls_map = {
    "ice": np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint8),  
    "snow": np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.uint8),  
    "debris": np.array([0, 0, 1, 0, 0, 0, 0, 0], dtype=np.uint8),  
    "firn": np.array([0, 0, 0, 1, 0, 0, 0, 0], dtype=np.uint8),  
    "shadow": np.array([0, 0, 0, 0, 1, 0, 0, 0], dtype=np.uint8),  
    "refrozen-like": np.array([0, 0, 0, 0, 0, 1, 0, 0], dtype=np.uint8),  
    "water": np.array([0, 0, 0, 0, 0, 0, 1, 0], dtype=np.uint8),  
    "cloud": np.array([0, 0, 0, 0, 0, 0, 0, 1], dtype=np.uint8),  
}
ftr_map = {
    "blue": [0],
    "green": [1],
    "red": [2],
    "nir": [3],
    "swir1": [4],
    "swir2": [5],
    "optical": [0, 1, 2, 3, 4, 5],
    "elevation": [6],
    "slope": [7],
    "shadow_mask": [8],
    "hillshade": [9],
    "proximity": [10],
}
default_batch_size = 512
default_label_smoothing = 0.1
# with open("perscenestats.pickle", "rb") as stats_src:
#     perscenestats = pickle.load(stats_src)
with open("globalminmax.pickle", "rb") as stats_src:
    mins, maxs = pickle.load(stats_src)

    
default_brightness_delta = 0.01
default_contrast_factor_range = 0.95, 1.05

    
def read_an_item(
    path, normalise=True, ftr_indices=slice(None, None), use_sample_weights=False, 
    class_weights=None, scene_weights=None,
):
    if use_sample_weights and (class_weights is None or scene_weights is None):
        raise ValueError("weight dicts must be provided if use_sample_weights.")
    filename = path.split(os.sep)[-1]
    filename = filename.split('.')[0]
    parts = filename.split('_')
    
    cls = parts[-1]
    glacier = parts[1]
    date = parts[2]
    weight = None
    if use_sample_weights:
        class_weight = class_weights[cls]
        scene_weight = scene_weights[(glacier, date)]
        weight = class_weight * scene_weight
    
    cls = cls_map[cls]
    patch_rst = rioxarray.open_rasterio(path)
    patch = patch_rst.data.astype(np.float32)
    patch = np.moveaxis(patch, 0, -1)
    patch = patch[..., ftr_indices]
    patch_rst.close()
    
    # if normalise:
    #     _, _, mean, std = perscenestats[glacier][date]
    #     patch = (patch - mean) / std
    if normalise:
        _mins = mins[ftr_indices]
        _maxs = maxs[ftr_indices]
        patch = (patch - _mins) / (_maxs - _mins)
    return patch, cls, weight


def random_augmentation(
    patch, label, weight=None, 
    brightness_delta=default_brightness_delta,
    contrast_factor_range=default_contrast_factor_range,
):
    patch = tf.image.random_flip_left_right(patch)
    patch = tf.image.random_flip_up_down(patch)
    k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
    patch = tf.image.rot90(patch, k)
    patch = tf.image.random_contrast(patch, contrast_factor_range[0], contrast_factor_range[1])
    patch = tf.image.random_brightness(patch, brightness_delta)
    return patch, label, weight


def label_to_float(
    patch, label, weight=None,
):
    label = tf.cast(label, tf.float32)
    return patch, label, weight


def label_smoothing(
    patch, label, weight=None, 
    label_smoothing=default_label_smoothing
):
    label = label * (1 - label_smoothing) + label_smoothing / tf.cast(tf.shape(label)[0], tf.float32)
    return patch, label, weight


def compute_class_weights(paths):
    weights = {}
    total = 0
    for path in paths:
        filename = path.split(os.sep)[-1]
        filename = filename.split('.')[0]
        cls = filename.split('_')[-1]
        if cls not in weights:
            weights[cls] = 0
        weights[cls] += 1
        total += 1
    for key in weights:
        weights[key] = total / weights[key]
    return weights


def compute_scene_weights(paths):
    weights = {}
    total = 0
    for path in paths:
        filename = path.split(os.sep)[-1]
        filename = filename.split('.')[0]
        parts = filename.split('_')
        glacier = parts[1]
        date = parts[2]
        key = (glacier, date)
        if key not in weights:
            weights[key] = 0
        weights[key] += 1
        total += 1
    for key in weights:
        weights[key] = total / weights[key]
    return weights


def get_folds_dataset(
    folder, folds, 
    features=None,
    use_augmentation=False, use_label_smoothing=False, shuffle=False,
    use_sample_weights=False,
    batch_size=default_batch_size,
    exclude=None, return_paths=False,
):
    if shuffle and return_paths:
        raise ValueError("shuffle + return_paths is not supported.")
    
    all_paths = []
    for fold in folds:
        all_paths.extend(sorted(glob.glob(f"{folder}/{fold}_*.tif")))
    if exclude is not None:
        all_paths = [_ for _ in all_paths if _ not in exclude]
    if shuffle:
        random.shuffle(all_paths)
    # all_paths = all_paths[::500] # for debugging

    ftr_indices = slice(None, None)
    if features:
        ftr_indices = []
        for feature in features:
            ftr_indices += ftr_map[feature]
    
    class_weights, scene_weights = None, None
    if use_sample_weights:
        class_weights = compute_class_weights(all_paths)
        scene_weights = compute_scene_weights(all_paths)
    
    dataset = []
    for path in tqdm(all_paths, desc="Reading data..."):
        dataset.append(read_an_item(
            path, ftr_indices=ftr_indices, use_sample_weights=use_sample_weights,
            class_weights=class_weights, scene_weights=scene_weights,
        ))
    x, y, weights = zip(*dataset)
    x, y, weights = list(x), list(y), list(weights)
    
    dataset = tf.data.Dataset.from_tensor_slices((x, y, weights))
    
    dataset = dataset.cache()
    dataset = dataset.map(label_to_float, num_parallel_calls=tf.data.AUTOTUNE)
    
    if use_label_smoothing:
        dataset = dataset.map(label_smoothing, num_parallel_calls=tf.data.AUTOTUNE)
    
    if use_augmentation:
        dataset = dataset.map(random_augmentation, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=dataset.cardinality() // 4, reshuffle_each_iteration=True)
    
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    if return_paths:
        return dataset, all_paths
    
    return dataset
    