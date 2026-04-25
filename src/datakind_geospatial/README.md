# **Pipelines**

This directory contains the main entry-point for Kedro pipelines used in this project. The different types of data processing pipelines are stored in the `pipelines`directory, which are all registered in `pipeline_registry.py`. 

## **Available pipelines**

As of know, the following pipelines have been implemented:

- `vi_preprocessing`: Unified pipeline form performing preprocessing steps on VI time-series datasets (NDVI, NDMI etc.)

A list of available pipelines can be found through `kedro registry list`. Furthermore, the pipelines and their connections with data sources and with each other can be visualized using `kedro viz`.

## **Running pipelines**

The `vi_preprocessing` pipeline preprocesses both NDVI and NDMI time-series data. Running the pipeline without node or parameter filters processes all configured VI datasets and all available regions:

```bash
kedro run --pipeline vi_preprocessing
```

To run only one VI product, select the corresponding node:

```bash
kedro run \
  --pipeline vi_preprocessing \
  --nodes preprocess_ndvi_timeseries_node
```

```bash
kedro run \
  --pipeline vi_preprocessing \
  --nodes preprocess_ndmi_timeseries_node
```

By default, `selected_regions` is `null` in `conf/base/parameters.yml`, which means all regions are processed. To run a specific region, override the relevant parameter for the VI product being processed:

```bash
kedro run \
  --pipeline vi_preprocessing \
  --nodes preprocess_ndvi_timeseries_node \
  --params "ndvi_preprocessing.selected_regions=['Kajiado_1']"
```

For multiple regions, pass a list:

```bash
kedro run \
  --pipeline vi_preprocessing \
  --nodes preprocess_ndvi_timeseries_node \
  --params "ndvi_preprocessing.selected_regions=['Kajiado_1','Kajiado_2']"
```

To run all regions explicitly, set the region filter to `null` or leave it unset:

```bash
kedro run \
  --pipeline vi_preprocessing \
  --nodes preprocess_ndvi_timeseries_node \
  --params "ndvi_preprocessing.selected_regions=null"
```

The same pattern applies to NDMI by replacing `ndvi_preprocessing` with `ndmi_preprocessing` and using `preprocess_ndmi_timeseries_node`.

The `selected_regions` filter controls which region folders are processed within a partitioned dataset. It does not select which nodes Kedro runs. Use `--nodes` when running only NDVI or only NDMI; otherwise, `kedro run --pipeline vi_preprocessing` will run both preprocessing nodes.
