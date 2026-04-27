"""
Feature engineering nodes
"""
from __future__ import annotations 

import logging
from collections.abc import Callable
from typing import Any

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sktime.datatypes import check_is_scitype
from sktime.transformations.panel.catch22 import Catch22

from .transformers import RemoveNanColumns

logger = logging.getLogger(__name__)

def reindex_data(
    train_data: dict[str, Callable[[], pd.DataFrame]],
    params: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Re-indexes training data to match SKTime data form and
    checks for missing uuid between training set and labels.

    Args:
        train_data (dict[str, Callable[[], pd.DataFrame]]): Kedro partitioned dataset containing data and label files
        params (dict[str, Any]): parameters from feature_engineering.reindex_train_data;
                                 parameters passed — 'data' (data file tag), 'label' (label file tag)
                                 and 'value_column' name of the column

    Returns:
        (pd.DataFrame, pd.DataFrame): re-indexed training and label dataframes
    """
    # Get required keys from params dict
    data_key = params.get("data")
    label_key = params.get("label")
    value_column = params.get("value_column", "ndvi")

    # Extract partitioned data using 'data_key' and 'label_key'
    df_data = train_data[data_key]()
    df_label = train_data[label_key]()
    df_data = df_data.copy()
    df_label = df_label.copy()

    df_data["date"] = pd.to_datetime(df_data["date"])

    # Find unnamed columns
    unnamed_cols = [col for col in df_data.columns if "Unnamed:" in col]
    if unnamed_cols:
        df_data.drop(columns=unnamed_cols, inplace=True)
    unnamed_label_cols = [col for col in df_label.columns if "Unnamed:" in col]
    if unnamed_label_cols:
        df_label.drop(columns=unnamed_label_cols, inplace=True)

    missing_uuids = list(set(df_data.uuid.unique()) - set(df_label.uuid.unique()))

    # Drop missing uuids if present
    df_data = df_data[~df_data.uuid.isin(missing_uuids)]
    df_label = df_label[df_label["uuid"].isin(df_data["uuid"].unique())]
    df_label = df_label.drop_duplicates(subset="uuid", keep="first").reset_index(drop=True)

    # Reindex the dataframe into one compatible with Sktime
    if value_column not in df_data.columns:
        raise KeyError(
            f"Value column '{value_column}' was not found in training dataframe columns: "
            f"{list(df_data.columns)}"
        )

    df_data_reindexed = (
        df_data.sort_values(["uuid", "date"])
        .set_index(["uuid", "date"])[[value_column]]
    )
    type_check = check_is_scitype(
        df_data_reindexed,
        scitype="Panel",
        return_metadata=True
    )

    if type_check[0]:
        logging.info(f"Dataframe has correct 'scitype': {type_check[2]['scitype']}")
        return df_data_reindexed, df_label
    else:
        raise TypeError(f"Dataframe does not have the correct 'scitype': {type_check[1]}")

def encode_classes(df_label: pd.DataFrame,  params: dict[str, Any]) -> pd.DataFrame:
    """
    If necessary, encodes classification classes in integers.

    Args:
        df_label (pd.DataFrame): dataframe with classification labels
        params (dict[str, Any]): parameters from feature_engineering.encode_classes;
                                 consists of label mapping

    Returns:
        (pd.DataFrame): label dataframe with encoded classes
    """
    df_label = df_label.copy()

    if "class_encoded" not in df_label.columns:
        df_label["class_encoded"] = df_label["class"].map(params)

    missing_class_mappings = df_label["class_encoded"].isna()
    if missing_class_mappings.any():
        missing_classes = sorted(df_label.loc[missing_class_mappings, "class"].unique().tolist())
        raise ValueError(f"Missing class encodings for labels: {missing_classes}")

    df_label["class_encoded"] = df_label["class_encoded"].astype(int)

    return df_label

def build_sklearn_pipeline(params: dict[str, Any] | None=None) -> Pipeline:
    """
    Returns SKLearn pipeline containing a sequence of feature transformations.
    Thie `.fit_transform()` and `.transform()` methods are only applied in the
    training pipeline once CV splits have been created.
    """
    params = params or {}

    return Pipeline([
        ("catch22", Catch22(catch24=params.get("catch24", True))),
        ("remove_nan", RemoveNanColumns()),
        ("imputer", SimpleImputer(strategy=params.get("impute_strategy", "median"))),
        ("scale", MinMaxScaler())
    ])
