# Polygon Labeling Widget

This Gradio app ports the notebook workflow from
`crop_classification/time_series_analyses/widget/label_maker.ipynb`.

It reads raw NDVI and NDMI CSVs from:

- `crop_classification/time_series_analyses/ndvi_series_raw`
- `crop_classification/time_series_analyses/ndmi_series_raw`

The app smooths each UUID time series, renders the selected polygon on Esri
World Imagery, and saves labels under `labeling_widget/labels`.

## Run

From the repository root:

```bash
python labeling_widget/app.py
```

The repository declares the required packages in `pyproject.toml`, including
`gradio`, `pandas`, `matplotlib`, and `scipy`. No dependencies are installed by
this app.

## Output

Labels are written as CSV files named:

```text
labeling_widget/labels/<region>_<tile>_labels.csv
```

Each row contains:

- `uuid`
- `class`
- `updated_at`
