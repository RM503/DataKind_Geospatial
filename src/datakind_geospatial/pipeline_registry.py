"""Project pipelines."""
from __future__ import annotations

from kedro.pipeline import Pipeline 

from .pipelines.feature_engineering.pipeline import create_feature_engineering_pipeline
from .pipelines.vi_preprocessing.pipeline import create_vi_preprocessing_pipeline

def register_pipelines() -> dict[str, Pipeline]:
    """Register available pipelines."""
    feature_engineering_pipeline = create_feature_engineering_pipeline()
    vi_preprocessing_pipeline = create_vi_preprocessing_pipeline()

    return {
        "feature_engineering": feature_engineering_pipeline,
        "vi_preprocessing": vi_preprocessing_pipeline,
        "__default__": vi_preprocessing_pipeline + feature_engineering_pipeline
    }