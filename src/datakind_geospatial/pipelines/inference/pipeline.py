from typing import Any

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import prepare_inference_data, run_inference


def create_inference_pipeline(**kwargs: dict[str, Any]) -> Pipeline:
    return pipeline([
        node(
            func=prepare_inference_data,
            inputs=[
                "ndvi_series_clean",
                "params:inference"
            ],
            outputs="ndvi_series_inference",
            name="prepare_inference_data_node"
        ),
        node(
            func=run_inference,
            inputs=[
                "nvdi_series_inference",
                "trained_classifier_pipeline",
                "params:inference"
            ],
            outputs="inference_predictions",
            name="run_inference_node"
        )
    ])
