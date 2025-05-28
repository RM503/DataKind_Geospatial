# Crop classification

## Analyses of NDVI time-series data

The `time_series_analyses` folder contains the code base for analyzing NDVI time-series data of delineated polygons which have been extracted using GEE. Since `segment-geospatial` is not able to distinguish between farms, fields and other types, NDVI time-series data can be used to train classification models for inference, allowing us to filter the large collection of polygons down to the ones that are of use.