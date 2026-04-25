"""Pipeline for VI time-series preprocessing"""
from typing import Any

from kedro.pipeline import Pipeline, node, pipeline 

from .nodes import preprocess_vi_timeseries

def create_vi_preprocessing_pipeline(**kwargs: dict[str, Any]) -> Pipeline:
    return pipeline([
        node(
            func=preprocess_vi_timeseries,
            inputs=[
                "ndvi_series_raw",
                "params:ndvi_preprocessing"
            ],
            outputs="ndvi_series_clean",
            name="preprocess_ndvi_timeseries_node"
        ),
        node(
            func=preprocess_vi_timeseries,
            inputs=[
                "ndmi_series_raw",
                "params:ndmi_preprocessing"
            ],
            outputs="ndmi_series_clean",
            name="preprocess_ndmi_timeseries_node"
        )
    ])