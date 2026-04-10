# Leveraging GIS and ML for improving circular economy prospects in Kenya

The effects of climate change have had a drastic impact on farmland productivity and crop yields across the world—especially for smallholder farms. Coupled with poor soil quality, farmers face an increasing challenge of staying competitive as markets become more dominated by commercial operations.

In this project, **DataKind** partnered with **Regen Organics** to support a data-driven study of farmlands across Kenya, with the goal of improving circular economy outcomes (e.g., better targeting and evaluation of soil and farm interventions).

## Overview

This repository brings together geospatial and machine learning pipelines used to:

- **Generate raster tiles** for target locations/regions from remote-sensing sources.
- **Segment farmland-like polygons** from raster tiles using a SAM-based segmentation workflow (via `segment-geospatial` / SamGeo).
- **Analyze vegetation index time series** (e.g., NDVI/NDMI/EVI) for delineated polygons and train **classification models** to filter polygons into useful categories (e.g., Farm vs Field vs Other/Tree).
- **Postprocess and export results** by merging predictions back into geospatial datasets (GeoPackage) for downstream analysis and mapping.

## Repository layout (what to look at)

- **`src/`**: the installable Python package (configured via `pyproject.toml` with `package-dir = "src"`).
  - **`src/generate_rasters/`**: raster generation pipeline (Google Earth Engine initialization + request building + GeoTIFF writing). Entry logic lives in `src/generate_rasters/main.py`.
  - **`src/segmentation/`**: segmentation pipeline and S3 I/O helpers (download tiles, run SAM-based segmentation, write outputs, upload artifacts). Core orchestration is in `src/segmentation/pipeline.py`.
  - **`src/configs/`** and **`configs/`**: configuration objects and example settings (e.g., S3 bucket names in `configs/settings.toml`).
- **`crop_classification/`**: analysis code and experiments for vegetation-index time-series and model training.
  - **`crop_classification/time_series_analyses/`**: end-to-end NDVI/VI time-series workflows (cleaning, transformations, MLflow experiments, inference scripts, and postprocessing utilities).
  - See **`crop_classification/README.md`** for more detail on the time-series classification workstream.
- **`samgeo_aws_ec2/`**: notes and scripts for running segmentation workloads on AWS EC2 (GPU instances). See `samgeo_aws_ec2/README.md`.

## Typical workflow (high level)

- **Raster generation**: start from a table/GeoDataFrame of target locations → generate GeoTIFF tiles per region.
- **Segmentation**: run SAM-based segmentation on tiles (often on GPU / EC2) → produce delineated polygons (GeoPackage/CSV/mask TIFF).
- **Time-series + classification**: compute vegetation index (VI) time series per polygon (GEE export) → clean/resample/smooth → extract features (e.g., Catch22) → train/track models (MLflow/Optuna) → batch inference.
- **Geospatial export**: merge predictions with polygon layers and attach time-series (nested format) for exploration and mapping.

## Getting started (development)

- **Python**: the project targets **Python 3.13+** (see `pyproject.toml`).
- **Install**: in editable mode from the repository root:

```bash
python -m pip install -e .
```

If you are using `uv`, you can also install/sync dependencies using your existing workflow (the repo includes a `uv.lock`).

## Notes

- Some workflows rely on external services and credentials (e.g., Google Earth Engine authentication, AWS S3 access, and MLflow tracking URIs). Check the relevant module READMEs and configs for required environment setup.