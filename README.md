# Automating glacier facies classification: pan-European dataset and deep learning baseline

[Konstantin A. Maslov](https://people.utwente.nl/k.a.maslov), [Thomas Schellenberger](https://www.mn.uio.no/geo/english/people/aca/geohyd/thosche/), [Prashant Pandit](https://www.eurac.edu/en/people/prashant-pandit), [Claudio Persello](https://people.utwente.nl/c.persello), [Alfred Stein](https://people.utwente.nl/a.stein)

[[`Paper`]()] [[`Datasets`](#datasets)] [[`BibTeX`](#citing)] 

<br/>

![results](assets/results.png)

Glacier facies play a critical role in understanding the mass balance of glaciers, offering insights into accumulation and melting processes. 
Large-scale mapping of glacier facies from satellite data is therefore essential for monitoring glacier response to climate change and informing climate policies. 
In this study, we present the largest glacier facies dataset ever compiled for Europe, comprising 31 glaciers, 92 Landsat and Sentinel-2 scenes, 137592 expert point labels and eight classes&mdash;five glacier facies (*ice*, *snow*, *debris*, *firn* and *refrozen-like*) and three miscellaneous classes (*shadow*, *water* and *cloud*)&mdash;encompassing a wide variety of surface conditions. 
A confident learning method pruned 16% of ambiguous expert labels overall. 
A compact and straightforward convolutional neural network reached a macro-average F1 score of 82% on the complete cleaned data or 74% on the full, unpruned data, and 82.3±10.5% glacier-wise. 
This performance remains consistent across different regions and sensors. 
When the facies products were regressed against World Glacier Monitoring Service records, they showed moderate, yet significant correlation with the surface mass balance measurements globally (*r* = 0.65, RMSE = 0.60 m w.e., where 1 m w.e. = 1000 kg m<sup>-2</sup> denotes metre water equivalent) and competitive correspondence for glacier-specific calibration (*r* = 0.79, RMSE = 0.28 m w.e.).
Overall, the dataset and baseline show that large-scale glacier facies classification can be achieved with high consistency. 
By providing both the dataset and baseline classification models, we aim to support the broader community in developing more advanced methods for glacier facies mapping to enhance our understanding of ongoing glacial changes. 

<br/>

## Datasets

The dataset can be accessed at [https://doi.org/10.5281/zenodo.18469893](https://doi.org/10.5281/zenodo.18469893). 
Place it in a separate folder and adjust the paths in `main.ipynb` accordingly. 


## Getting started

We recommend using the [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/) Python distributions. 
After installing one of them, one can use the `conda` package manager to install the required libraries in a new environment called `massive-tf` and activate it by running

```
conda create -n massive-tf "tensorflow<2.16" cleanlab h5py scikit-learn rioxarray geopandas jupyterlab tqdm matplotlib -c conda-forge
conda activate massive-tf
```

We tested this configuration on Ubuntu 22.04 (see `env.yml` for the exported environment). 
We also expect it to work on any modern Linux distribution or Windows, given properly configured NVIDIA GPU drivers.

<br/>

To reproduce the results presented in the manuscript, follow [`main.ipynb`](main.ipynb) for the main training-evaluation cycles and [`confidence_calibration.ipynb`](confidence_calibration.ipynb) for the predictive confidence calibration. 
The pretrained models are available in `weights/<FOLD>_cnn<before|after>confidencelearning_weights.h5` (the best feature set) or in `weights/<FEATURESET>/<FOLD>_cnn<before|after>confidencelearning_weights.h5`. 
Similarly, the training logs are stored in `logs/...`.

<br/>

Applying a model to a complete scene can be achieved by means of `apply.py`. 
The usage is as follows:
```
usage: apply.py [-h] [--dem_path DEM_PATH] [--slope_path SLOPE_PATH] [--shadow_path SHADOW_PATH] [--hillshade_path HILLSHADE_PATH] [--proximity_path PROXIMITY_PATH]
                [--confidence_path CONFIDENCE_PATH] [--calibration_model CALIBRATION_MODEL] [--no_smoothing]
                model_name raster_path output_path outlines_path

positional arguments:
  model_name            Model name
  raster_path           Raster path (.tif)
  output_path           Output path (.tif)
  outlines_path         Outlines path (.shp/.geojson)

options:
  -h, --help            show this help message and exit
  --dem_path DEM_PATH   Elevation path (.tif)
  --slope_path SLOPE_PATH
                        Slope path (.tif)
  --shadow_path SHADOW_PATH
                        Shadow mask (0/1) path (.tif)
  --hillshade_path HILLSHADE_PATH
                        Hillshade path (.tif)
  --proximity_path PROXIMITY_PATH
                        Proximity path (.tif)
  --confidence_path CONFIDENCE_PATH
                        Output confidence path (.tif)
  --calibration_model CALIBRATION_MODEL
                        Confidence calibration model (.pickle)
  --no_smoothing        Turn off mode smoothing of results
```
Model name follows `<FOLD>_cnn<before|after>confidencelearning`. 
Make sure that the number of provided features and the number of features expected by the model are aligned, as well as that all input raster are defined on identical 10 m grids.


## License

This software is licensed under the [GNU General Public License v2](LICENSE).


## Citing

To cite the paper/repository, please use the following bib entry. 

```
@misc{massiveteam_glacierfaciesclassification_2026,
  author       = {Maslov, K. A. and Schellenberger, T. and Pandit, P. and Persello, C. and Stein, A.},
  title        = {Automating glacier facies classification: pan-European dataset and deep learning baseline},
  year         = {2026},
  howpublished = {EarthArXiv preprint},
  url          = {https://eartharxiv.org/repository/view/11580/},
  doi          = {10.31223/X5QN1V},
  note         = {Accessed: 2026-01-27}
}
```
