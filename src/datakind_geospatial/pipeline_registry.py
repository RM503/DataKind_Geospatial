"""Project pipelines."""
from __future__ import annotations

from kedro.pipeline import Pipeline

from .pipelines.feature_engineering.pipeline import create_feature_engineering_pipeline
from .pipelines.inference.pipeline import create_inference_pipeline
from .pipelines.training.pipeline import create_training_pipeline
from .pipelines.vi_preprocessing.pipeline import create_vi_preprocessing_pipeline

def register_pipelines() -> dict[str, Pipeline]:
    """Register available pipelines."""
    feature_engineering_pipeline = create_feature_engineering_pipeline()
    inference_pipeline = create_inference_pipeline()
    training_pipeline = create_training_pipeline()
    vi_preprocessing_pipeline = create_vi_preprocessing_pipeline()
    classification_training_pipeline = feature_engineering_pipeline + training_pipeline
    training_inference_pipeline = classification_training_pipeline + inference_pipeline

    return {
        "feature_engineering": feature_engineering_pipeline,
        "inference": inference_pipeline,
        "training": classification_training_pipeline,
        "classification_training": classification_training_pipeline,
        "training_inference": training_inference_pipeline,
        "vi_preprocessing": vi_preprocessing_pipeline,
        "__default__": classification_training_pipeline
    }
