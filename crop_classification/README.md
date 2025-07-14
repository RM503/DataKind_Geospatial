# Crop classification

## Analyses of NDVI time-series data

The `time_series_analyses` folder contains the code base for analyzing Vegetation Index (VI) time-series data of delineated polygons which have been extracted using GEE. Since `segment-geospatial` is not able to distinguish between farms, fields and other types, NDVI time-series data can be used to train classification models for inference, allowing us to filter the large collection of polygons down to the ones that are of use. The folder contains data preprocessing, transformation and post processing scripts in the `data_processing` folder. ML classification experiment codes are stored in `mlflow_experiments`. 