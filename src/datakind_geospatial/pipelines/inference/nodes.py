"""
Local inference nodes.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline
from tqdm import tqdm

logger = logging.getLogger(__name__)


def prepare_inference_data(
    inference_data: dict[str, Callable[[], pd.DataFrame]],
    params: dict[str, Any]
) -> pd.DataFrame:
    """
    Prepares cleaned NVDI data for inference.
    """
    value_column = params.get("value_column", "ndvi")
    output_id_column = params.get("output_id_column", "uuid")
    selected_partitions = params.get("selected_partitions")
    selected_regions = params.get("selected_regions")

    inference_data_partitions = []

    for partition_key, partition_load_fn in tqdm(
        inference_data.items(),
        total=len(inference_data),
        desc=f"Preparing {value_column.upper()} for inference"
    ):
        if selected_partitions and partition_key not in selected_partitions:
            continue
        if selected_regions and not any(
            partition_key.startswith(f"{region}/" for region in selected_regions)
        ):
            continue
        # Load the dataframe for given partition key
        df = partition_load_fn()

        unnamed_cols = [col for col in df.columns if "Unnamed:" in str(col)]
        if unnamed_cols:
            df = df.drop(columns=unnamed_cols)

        df["date"] = pd.to_datetime(df["date"])
        df_transformed = (
            df.sort_values([output_id_column, "date"])
              .set_index([output_id_column, "date"])[[value_column]]
        )

        inference_data_partitions.append(df_transformed)

    return pd.concat(inference_data_partitions).sort_index()


def run_inference(
    prepared_inference_data: pd.DataFrame,
    trained_classifier: Pipeline,
    params: dict[str, Any]
) -> pd.DataFrame:
    output_id_column = params.get("output_id_column", "uuid")

    uuid_order = prepared_inference_data.index.get_level_values(output_id_column).drop_duplicates()
    predictions = trained_classifier.predict(prepared_inference_data)

    # Prepare prediction output
    output = pd.DataFrame({
        params.get(output_id_column, "uuid"): uuid_order,
        params.get("prediction_column", "prediction"): predictions
    })

    class_decodings = params.get("class_decodings")
    if class_decodings:
        output[params.get("decoded_prediction_column", "prediction_decoded")] = (
            output[params.get("prediction_column", "prediction")]
            .map(class_decodings)
        )

    return output
