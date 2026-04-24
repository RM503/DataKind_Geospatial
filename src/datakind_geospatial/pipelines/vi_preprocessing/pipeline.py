"""Pipeline for VI time-series preprocessing"""
from typing import Any

from kedro.pipeline import Pipeline, node 

from .nodes import preprocess_vi_timeseries

def create_vi_preprocessing_pipeline(**kwargs: dict[str, Any]) -> Pipeline:
    return Pipeline([
        node(
            func=preprocess_vi_timeseries,
            inputs=["ndvi_series_raw.Trans_Nzoia_1", "params:ndvi_preprocessing"],
            outputs="ndvi_series_clean.Trans_Nzoia_1"
        )
    ])