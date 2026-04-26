from __future__ import annotations 

import logging
from collections.abc import Callable
from typing import Any

import pandas as pd
from sktime.datatypes import check_is_scitype

logger = logging.getLogger(__name__)

def reindex_data(
    train_data: dict[str, Callable[[], pd.DataFrame]],
    params: dict[str, Any]
) -> pd.DataFrame:
    data_key = params.get("data")
    label_key = params.get("label")

    df_data = train_data[data_key]()
    df_data.date = pd.to_datetime(df_data.date)
    df_label = train_data[label_key]()

    # Find unnamed columns
    unnamed_cols = [col for col in df_data.columns if "Unnamed:" in col]
    if unnamed_cols:
        df_data.drop(columns=unnamed_cols, inplace=True)

    missing_uuids = list(set(df_data.uuid.unique()) - set(df_label.uuid.unique()))

    # Drop missing uuids if present
    df_data = df_data[~df_data.uuid.isin(missing_uuids)]

    # Reindex the dataframe into one compatible with Sktime
    df_reindexed = df_data.set_index(["uuid", "date"])[["ndvi"]]
    type_check = check_is_scitype(
        df_reindexed,
        scitype="Panel",
        return_metadata=True
    )

    if type_check[0]:
        logging.info(f"Dataframe has correct 'scitype': {type_check[2]['scitype']}")
        return df_reindexed
    else:
        raise TypeError(f"Dataframe does not have the correct 'scitype': {type_check[1]}")