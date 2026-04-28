from typing import Any

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import train_classifier

def create_training_pipeline(**kwargs: dict[str, Any]) -> Pipeline:
    return pipeline(
        [
            node(
                func=train_classifier,
                inputs=[
                    "train_data_reindexed",
                    "train_label_encoded",
                    "params:training",
                    "params:feature_engineering.sklearn_transformation_pipeline",
                ],
                outputs=["training_summary", "trained_classifier_pipeline"],
                name="train_timeseries_classifier_node",
            )
        ]
    )
