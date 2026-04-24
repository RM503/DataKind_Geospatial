# datakind-geospatial

[![Python](https://img.shields.io/badge/python-3.13%2B-blue.svg)](https://www.python.org/)
[![Package manager: uv](https://img.shields.io/badge/package%20manager-uv-6e56cf.svg)](https://github.com/astral-sh/uv)
[![Framework: Kedro](https://img.shields.io/badge/framework-Kedro-1.3.1-ffc900.svg)](https://kedro.org/)
[![MLflow](https://img.shields.io/badge/tracking-MLflow-0194E2.svg)](https://mlflow.org/)
[![Status](https://img.shields.io/badge/status-in%20progress-orange.svg)](#roadmap)

Geospatial and machine learning workflows for DataKind's Kenya farmland analysis work, now being reorganized into a Kedro project.

This README is intentionally incomplete. It is meant to give the team a usable project entry point while the new Kedro structure and pipeline docs are still being filled in.

## Background

This repository supports geospatial analysis and modeling work related to farmland identification, vegetation-index analysis, and downstream classification workflows tied to circular economy outcomes in Kenya.

Historically, the repo has contained a mix of standalone scripts, notebooks, experiment folders, and workflow-specific modules. It now also includes a Kedro project under `src/datakind_geospatial/` with shared config in `conf/`.

## What is in the repo

The current codebase appears to break down into a few main workstreams:

- `src/generate_rasters/`: utilities for building remote-sensing requests and generating raster assets.
- `src/segmentation/`: SAM/SamGeo-based field segmentation pipeline, including S3 I/O and SageMaker-oriented processing code.
- `crop_classification/`: vegetation-index time-series analysis, feature engineering, and crop or land-type classification experiments.
- `labeling_widget/`: Gradio-based polygon labeling workflow for time-series review.
- `samgeo_aws_ec2/`: older AWS EC2 notes and scripts for segmentation workloads.
- `src/datakind_geospatial/`: Kedro package, CLI entrypoint, settings, pipeline registry, and new pipeline namespace.

## Kedro addition

The repository now includes Kedro project scaffolding:

- `src/datakind_geospatial/__main__.py` exposes `datakind-geospatial` and `python -m datakind_geospatial`.
- `src/datakind_geospatial/pipeline_registry.py` uses `find_pipelines()` and assembles a `__default__` pipeline.
- `src/datakind_geospatial/pipelines/` is the new home for Kedro-managed pipelines.
- `conf/base/` and `conf/local/` provide shared vs local configuration split.
- `conf/base/catalog.yml` already defines partitioned datasets for raw and clean vegetation-index series.

At the moment, the generated `data_processing` Kedro pipeline is still effectively a placeholder:

- `src/datakind_geospatial/pipelines/data_processing/pipeline.py` returns an empty `Pipeline([])`.
- `conf/base/parameters_data_processing.yml` is still boilerplate.

That means the Kedro structure is present, but the migration of legacy workflows into Kedro pipelines is still in progress.

## Project structure

```text
.
├── conf/                         # Kedro configuration
│   ├── base/
│   └── local/
├── configs/                      # Non-Kedro project configs used by legacy workflows
├── crop_classification/          # Time-series analysis and ML experiments
├── docker/                       # Container assets for processing jobs
├── jobs/                         # Job entrypoints, including segmentation processing jobs
├── labeling_widget/              # Gradio labeling application
├── notebooks/                    # Exploratory and workflow notebooks
├── samgeo_aws_ec2/               # EC2-era segmentation scripts and notes
├── src/
│   ├── classification/
│   ├── common/
│   ├── configs/
│   ├── data/
│   ├── datakind_geospatial/      # Kedro package
│   ├── generate_rasters/
│   └── segmentation/
├── pyproject.toml
└── uv.lock
```

## Setup

### Requirements

- Python `3.13+`
- `uv` recommended for environment and dependency management
- Credentials for whichever workflows you plan to run:
  - Google Earth Engine
  - AWS / S3 / SageMaker
  - MLflow backend
  - Supabase

### Install

```bash
uv sync
```

If you prefer plain `pip`:

```bash
python -m pip install -e .
```

To include notebook dependencies:

```bash
uv sync --extra notebooks
```

## Running the project

### Kedro commands

Run the Kedro project through the generated package entrypoint:

```bash
uv run datakind-geospatial run
```

Or:

```bash
uv run python -m datakind_geospatial run
```

Useful Kedro commands while the migration is underway:

```bash
uv run datakind-geospatial registry list
uv run kedro catalog list
uv run kedro viz
```

### Other workflows

Some existing workflows are still script- or notebook-driven rather than Kedro-driven:

- Segmentation job submission: see `jobs/segmentation/run_processing_job.py` and [src/segmentation/README.md](src/segmentation/README.md)
- Crop classification experiments: see [crop_classification/README.md](crop_classification/README.md)
- Labeling app: see [labeling_widget/README.md](labeling_widget/README.md)

## Current dependencies

The project currently pulls together tooling across:

- Kedro, Kedro Viz, Kedro MLflow, and Kedro SageMaker
- GeoPandas, Rasterio, Shapely, Fiona, PyProj
- Earth Engine and Sentinel Hub clients
- PyTorch and SAM-adjacent segmentation workflows
- Scikit-learn, LightGBM, XGBoost, Optuna, sktime, statsmodels
- MLflow for experiment tracking

See [pyproject.toml](pyproject.toml) for the authoritative dependency list.

## Configuration notes

- Shared Kedro config lives in `conf/base/`.
- Local and sensitive config belongs in `conf/local/` and should not be committed.
- There is also a legacy `configs/` directory used by existing non-Kedro workflows.
- `conf/base/catalog.yml` currently includes partitioned dataset definitions for raw and cleaned time-series data.

## Known gaps

- Pipeline-level documentation is still incomplete.
- The new Kedro `data_processing` pipeline is scaffolded but not implemented yet.
- The relationship between legacy scripts and future Kedro pipelines still needs to be documented clearly.
- Testing, linting, and CI instructions are not documented here yet.
- Example local setup for credentials is still missing.

## Roadmap

- [ ] Document how legacy workflows map into Kedro pipelines
- [ ] Implement the first non-empty Kedro pipeline
- [ ] Add dataset catalog and parameter docs
- [ ] Add reproducible development and testing commands
- [ ] Add architecture notes for raster generation, segmentation, and classification

## References

- [Kedro configuration notes](conf/README.md)
- [Segmentation workflow notes](src/segmentation/README.md)
- [Crop classification notes](crop_classification/README.md)
- [Labeling widget notes](labeling_widget/README.md)

## TODO

- Add a clearer end-to-end workflow diagram
- Add data folder conventions
- Add sample commands for each major workflow
- Add contributor guidance
