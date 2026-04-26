from typing import Any 

from kedro.pipeline import Pipeline, node, pipeline 

from .nodes import encode_classes, reindex_data

def create_feature_engineering_pipeline(**kwargs: dict[str, Any]) -> Pipeline:
    return pipeline([
        node(
            func=reindex_data,
            inputs=[
                "train_data",
                "params:feature_engineering.reindex_train_data"
            ],
            outputs=["train_data_reindexed", "train_label"]
        ),
        node(
            func=encode_classes,
            inputs=[
                "train_label",
                "params:feature_engineering.class_encodings"
            ],
            outputs=["train_label_encoded"]
        )
    ])