# **datakind-geospatial**

[![Python](https://img.shields.io/badge/python-3.13%2B-blue.svg)](https://www.python.org/)
[![Package manager: uv](https://img.shields.io/badge/package%20manager-uv-6e56cf.svg)](https://github.com/astral-sh/uv)
[![MLflow](https://img.shields.io/badge/tracking-MLflow-0194E2.svg)](https://mlflow.org/)
[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)


## **Introduction**
This repository contains eospatial and machine learning workflows for DataKind Singapore's project with Kenya-based Regen Organics. It includes legacy experiment code and a Kedro project for vegetation-index preprocessing, time-series feature engineering, and farmland classification.

## **Project Layout**

```text
.
├── conf/                         # Kedro catalog, parameters, credentials, logging
│   ├── base/
│   └── local/
├── crop_classification/          # Legacy time-series classification experiments
├── data/                         # Kedro data layers, not committed
├── notebooks/                    # Exploratory notebooks
├── src/
│   ├── classification/           # Legacy/non-Kedro classification helpers
│   ├── datakind_geospatial/      # Kedro project package
│   ├── generate_rasters/         # Raster-generation utilities
│   └── segmentation/             # Segmentation workflow code
├── pyproject.toml
└── uv.lock
```

## **Setup**

Requirements:

- Python `3.13+`
- `uv`
- Workflow-specific credentials as needed for Earth Engine, AWS/S3/SageMaker, Supabase, or remote MLflow

Install the project:

```bash
uv sync
```

Install notebook extras:

```bash
uv sync --extra notebooks
```

Run Kedro commands through the project entrypoint:

```bash
uv run datakind-geospatial --help
```

Equivalent module entrypoint:

```bash
uv run python -m datakind_geospatial --help
```

## **Kedro Pipelines**

The active Kedro pipelines are registered in `src/datakind_geospatial/pipeline_registry.py`. As of now, the following pipelines are available in the project —

| Pipeline | Command | Purpose |
| --- | --- | --- |
| `vi_preprocessing` | `uv run datakind-geospatial run --pipeline vi_preprocessing` | Cleans raw NDVI and NDMI partitioned time series. |
| `feature_engineering` | `uv run datakind-geospatial run --pipeline feature_engineering` | Loads training time series and labels, reindexes panel data, and encodes classes. |
| `training` | `uv run datakind-geospatial run --pipeline training` | Runs feature engineering plus classifier training. |
| `classification_training` | `uv run datakind-geospatial run --pipeline classification_training` | Alias for the full classification training workflow. |
| `__default__` | `uv run datakind-geospatial run` | Same as `classification_training`. |

Useful inspection commands:

```bash
uv run datakind-geospatial registry list
uv run kedro catalog list
uv run kedro viz
```

## **Data Layout**

The catalog is defined in `conf/base/catalog.yml`.

Expected inputs:

```text
data/01_raw/ndvi_series_raw/*.csv
data/01_raw/ndmi_series_raw/*.csv
data/04_train/*.csv
```

Training currently expects partition names configured in `conf/base/parameters.yml`:

```yaml
feature_engineering:
  reindex_train_data:
    data: Trans_Nzoia_1_ndvi_train
    label: Trans_Nzoia_1_label_train
    value_column: ndvi
```

Outputs:

```text
data/02_clean/ndvi_series_clean/
data/02_clean/ndmi_series_clean/
data/06_models/trained_classifier_pipeline.pkl
data/08_reporting/training_summary.json
```

## **VI Preprocessing Workflow**

Run:

```bash
uv run datakind-geospatial run --pipeline vi_preprocessing
```

Configuration lives under:

```yaml
ndvi_preprocessing:
ndmi_preprocessing:
```

Both inherit defaults from `vi_preprocessing_defaults` in `conf/base/parameters.yml`. The pipeline supports partition and region filtering through:

```yaml
selected_partitions: null
selected_regions: null
```

Set either value to a list when you want to process only a subset.

## **Classification Training Workflow**

Run the full training workflow:

```bash
uv run datakind-geospatial run --pipeline classification_training
```

This workflow performs:

1. Load partitioned training data and labels from `data/04_train`.
2. Reindex the time-series frame to an sktime-compatible panel index.
3. Encode labels from configured class names to integers.
4. Extract Catch22/Catch24 features.
5. Run stratified cross-validation.
6. Optionally run Optuna hyperparameter search.
7. Fit a final sklearn pipeline on all training data.
8. Save local Kedro outputs and log MLflow metrics/artifacts/model.

Current class encoding:

```yaml
Farm: 0
Field: 1
Other: 2
Tree: 3
```

The training code aligns labels to the feature panel UUID order before CV and final fitting. This is important because the panel can be sorted or filtered independently from the label CSV.

## **Model Configuration**

Model parameters live under:

```yaml
training:
  active_model: xgboost
  classifiers:
    xgboost:
    lightgbm:
```

Hyperparameter search is controlled by:

```yaml
training:
  hyperparameter_search:
    enabled: true
    n_trials: 10
```

Disable search when you want to run the base model parameters directly:

```yaml
training:
  hyperparameter_search:
    enabled: false
```

## **MLflow Workflow**

MLflow configuration lives under:

```yaml
training:
  mlflow:
    enabled: true
    tracking_uri: http://localhost:5000
    experiment_name: timeseries_classification_local
    artifact_path: timeseries_classifier
    registered_model_name: null
    log_model: true
```

Start a local MLflow server from the repo root:

```bash
uv run mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
```

Then run training:

```bash
uv run datakind-geospatial run --pipeline classification_training
```

What gets logged:

- Optuna trial runs log trial parameters and `trial.selection_metric`.
- The final training run logs summary metrics, `training_summary.json`, `confusion_matrix.png`, and `precision_recall_curves.png`.
- The final fitted sklearn pipeline is logged as an MLflow model named by `artifact_path`.

With MLflow 3, logged models may appear under the run's **Models / Outputs** section instead of as a normal folder in the run artifact tree. On disk they can appear under:

```text
mlruns/<experiment_id>/models/<model_id>/artifacts/model.pkl
```

If Kedro or `kedro-mlflow` already has an active parent run, final artifacts may be logged to that active Kedro experiment while Optuna trials appear in `training.mlflow.experiment_name`. If the UI shows only `xgboost-trial-*` runs, check the `datakind_geospatial` experiment for the parent `classification_training` run.

To register the model in the MLflow Model Registry, set:

```yaml
registered_model_name: timeseries_classifier
```

and rerun training.

## **Common Workflows**

Run the default classification workflow:

```bash
uv run datakind-geospatial run
```

Run only feature engineering:

```bash
uv run datakind-geospatial run --pipeline feature_engineering
```

Run training without MLflow:

```yaml
training:
  mlflow:
    enabled: false
```

Then:

```bash
uv run datakind-geospatial run --pipeline classification_training
```

Run a faster local smoke test by lowering trial count:

```yaml
training:
  hyperparameter_search:
    enabled: true
    n_trials: 1
```

## **Development Checks**

Compile the Kedro training package:

```bash
uv run python -m compileall -q src/datakind_geospatial/pipelines/training
```

Run linting on changed files:

```bash
uv run ruff check src/datakind_geospatial conf
```

Check git state before committing:

```bash
git status --short
```

## Legacy Workflows

Some workflows are still maintained outside Kedro:

- Legacy classification experiments: `crop_classification/time_series_analyses/mlflow_experiments/classification`
- Segmentation workflow: `src/segmentation/`
- Raster generation utilities: `src/generate_rasters/`
- Exploratory workflows: `notebooks/`

Use the Kedro pipelines for reproducible preprocessing and training where possible. Use legacy scripts as references for older experiments and comparison runs.

## **Configuration Notes**

- Shared configuration belongs in `conf/base/`.
- Local credentials and machine-specific config belong in `conf/local/`.
- Do not commit secrets.
- The authoritative dependency list is `pyproject.toml`.

## **Troubleshooting**

No final plots in `timeseries_classification_local`:

- Check whether the final run is in the `datakind_geospatial` experiment.
- Trial runs named `xgboost-trial-*` do not log plots by default.
- Final plots are named `confusion_matrix.png` and `precision_recall_curves.png`.

Model not visible in the artifact tree:

- In MLflow 3, check the run's **Models / Outputs** section.
- On disk, check `mlruns/<experiment_id>/models/<model_id>/artifacts/model.pkl`.
- Set `registered_model_name` if you need a Model Registry entry.

## **References**

- `conf/base/catalog.yml`
- `conf/base/parameters.yml`
- `src/datakind_geospatial/pipeline_registry.py`
- `src/datakind_geospatial/pipelines/`
- `src/segmentation/README.md`
