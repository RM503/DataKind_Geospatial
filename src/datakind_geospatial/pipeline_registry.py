"""Project pipelines."""
from __future__ import annotations

from kedro.pipeline import Pipeline 

from .pipelines.vi_preprocessing.pipeline import create_vi_preprocessing_pipeline

def register_pipelines() -> dict[str, Pipeline]:
    """Register available pipelines."""
    vi_preprocessing_pipeline = create_vi_preprocessing_pipeline()

    return {
        "vi_preprocessing": vi_preprocessing_pipeline
    }