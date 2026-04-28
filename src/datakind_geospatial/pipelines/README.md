# Pipelines

This directory contains the Kedro pipelines for vegetation-index preprocessing, feature engineering, and classification training.

Pipelines are registered in `src/datakind_geospatial/pipeline_registry.py`.

## Available Pipelines

| Pipeline | Purpose |
| --- | --- |
| `vi_preprocessing` | Cleans raw NDVI and NDMI partitioned time-series datasets. |
| `feature_engineering` | Loads training data and labels, reindexes the time-series panel, and encodes classes. |
| `training` | Runs feature engineering plus classifier training. |
| `classification_training` | Alias for the full feature-engineering and training workflow. |
| `__default__` | Same as `classification_training`. |

List available pipelines:

```bash
uv run datakind-geospatial registry list
```

Visualize pipeline structure:

```bash
uv run kedro viz
```

## VI Preprocessing

Run both NDVI and NDMI preprocessing:

```bash
uv run datakind-geospatial run --pipeline vi_preprocessing
```

Run only NDVI:

```bash
uv run datakind-geospatial run \
  --pipeline vi_preprocessing \
  --nodes preprocess_ndvi_timeseries_node
```

Run only NDMI:

```bash
uv run datakind-geospatial run \
  --pipeline vi_preprocessing \
  --nodes preprocess_ndmi_timeseries_node
```

Configuration lives in `conf/base/parameters.yml`:

```yaml
ndvi_preprocessing:
ndmi_preprocessing:
```

Both inherit defaults from `vi_preprocessing_defaults`.

Use `selected_partitions` or `selected_regions` to process subsets:

```bash
uv run datakind-geospatial run \
  --pipeline vi_preprocessing \
  --nodes preprocess_ndvi_timeseries_node \
  --params "ndvi_preprocessing.selected_regions=['Kajiado_1']"
```

The `selected_regions` parameter filters region folders inside the partitioned dataset. It does not select Kedro nodes. Use `--nodes` to run only NDVI or only NDMI.

## Feature Engineering

Run feature engineering only:

```bash
uv run datakind-geospatial run --pipeline feature_engineering
```

The pipeline expects training partitions in `data/04_train`, configured in `conf/base/parameters.yml`:

```yaml
feature_engineering:
  reindex_train_data:
    data: Trans_Nzoia_1_ndvi_train
    label: Trans_Nzoia_1_label_train
    value_column: ndvi
```

Feature engineering does two things:

1. Reindexes time-series data to an sktime-compatible panel index: `uuid`, `date`.
2. Encodes class labels using configured class mappings.

Current class mapping:

```yaml
Farm: 0
Field: 1
Other: 2
Tree: 3
```

## Classification Training

Run the full classification workflow:

```bash
uv run datakind-geospatial run --pipeline classification_training
```

The default Kedro run is equivalent:

```bash
uv run datakind-geospatial run
```

Training performs:

1. Feature engineering.
2. Catch22/Catch24 time-series feature extraction.
3. Stratified cross-validation.
4. Optional Optuna hyperparameter search.
5. Final model fitting on all training data.
6. Kedro output persistence.
7. MLflow logging.

The training code aligns labels to the feature panel UUID order before cross-validation and final fitting. This prevents silent feature/label misalignment when time-series rows are sorted or filtered separately from the label file.

## Model and Search Configuration

Training configuration lives under:

```yaml
training:
  active_model: xgboost
```

Supported model config sections:

```yaml
training:
  classifiers:
    xgboost:
    lightgbm:
```

Hyperparameter search:

```yaml
training:
  hyperparameter_search:
    enabled: true
    n_trials: 10
```

Disable HPO for a direct base-parameter run:

```yaml
training:
  hyperparameter_search:
    enabled: false
```

## Outputs

Kedro catalog outputs:

```text
data/06_models/trained_classifier_pipeline.pkl
data/08_reporting/training_summary.json
```

MLflow logs:

- Trial runs named like `xgboost-trial-0` log trial params and `trial.selection_metric`.
- The final training run logs aggregate metrics, `training_summary.json`, `confusion_matrix.png`, and `precision_recall_curves.png`.
- The final fitted sklearn pipeline is logged as an MLflow model.

With MLflow 3, logged models may appear under the run's **Models / Outputs** section rather than as a regular artifact folder. On disk, model files can appear under:

```text
mlruns/<experiment_id>/models/<model_id>/artifacts/model.pkl
```

If the MLflow UI shows only `xgboost-trial-*` runs in `timeseries_classification_local`, check the `datakind_geospatial` experiment for the parent `classification_training` run. Kedro or `kedro-mlflow` can create an active parent run, and the final plots/model may be logged there while trial runs appear in the configured training experiment.

## Data Catalog

Relevant catalog entries live in `conf/base/catalog.yml`.

Inputs:

```text
ndvi_series_raw
ndmi_series_raw
train_data
```

Outputs:

```text
ndvi_series_clean
ndmi_series_clean
training_summary
trained_classifier_pipeline
```

## Troubleshooting

Low classification F1:

- Confirm `feature_engineering.reindex_train_data.value_column` matches the input CSV.
- Confirm training labels contain every UUID in the feature panel.
- Confirm `training.target_label` is the class you intend to evaluate.

Missing plots in MLflow:

- Trial runs do not log plots by default.
- Final plots are logged on the final parent training run.
- Check both `timeseries_classification_local` and `datakind_geospatial` experiments.

Missing model in MLflow:

- Check the run's **Models / Outputs** section.
- Check `mlruns/<experiment_id>/models/<model_id>/artifacts/model.pkl`.
- Set `training.mlflow.registered_model_name` if you need a Model Registry entry.
