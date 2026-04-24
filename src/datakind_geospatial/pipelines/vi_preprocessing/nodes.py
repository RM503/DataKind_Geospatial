"""Data preprocessing nodes"""
from __future__ import annotations 

import logging
from typing import Any, Callable

import numpy as np 
import pandas as pd
import pandera as pa
from scipy.signal import savgol_filter
from sklearn.ensemble import IsolationForest

from datakind_geospatial.data_validators.vi_validator import VIDataValidator

logger = logging.getLogger(__name__)

def fill_dates(row: pd.Series) -> pd.Series:
    if pd.isna(row.iloc[-1]) and not pd.isna(row.iloc[0]):
        return row.ffill()
    return row.bfill()

def find_outliers(
    col: pd.Series, 
    n_estimators: int,
    contamination: float, 
    random_state: int
) -> np.ndarray:
    """Detects outliers on time-series using Isolation Forest method."""
    x = col.to_numpy().reshape(-1, 1)

    model = IsolationForest(
        n_estimators=n_estimators, 
        contamination=contamination,
        random_state=random_state
    )

    y_pred = model.fit(x)
    return y_pred

def date_resample(df: pd.DataFrame, vi_column: str, resample_freq: str) -> pd.DataFrame:
    if df["date"].dtype != "datetime64[ns]":
        df["date"] = pd.to_datetime(df["date"], coerce=True)

    # Check if there are multiple sampling intervals in the data
    # Resample to make uniform
    if len(df["date"].diff().value_counts()) > 1:
        logger.info(f"Multiple data sampling intervals found. Resampling to {resample_freq}.")

        df = (
            df.set_index("date").resample(resample_freq)
              .asfreq()
        )

        # Fill the missing uuid at resampled points
        df[vi_column] = df[vi_column].interpolate()
        df["uuid"] = df["uuid"].fillna(df["uuid"].mode())[0]

        return df.reset_index()
    return df

def resample_vi_group(
    group: pd.DataFrame,
    vi_column: str,
    uuid: str,
    resample_freq: str
) -> pd.DataFrame:
    out = group.set_index("date").resample(resample_freq).asfreq()
    out[vi_column] = out[vi_column].interpolate(method="linear")
    out["uuid"] = uuid 

    return out.reset_index()

def clean_vi_series(
    df: pd.DataFrame,
    vi_column: str,
    fill_method: str,
    resample_date: bool,
    resample_freq: str,
    smoothing_window: int,
    smoothing_polygon_order: int,
    n_estimators: int,
    outlier_contamination: float,
    random_state: int
) -> pd.DataFrame:
    """
    Restructures the VI row-major table by melting the dataframe, stacking time-series
    vertically for each uuid.

    Args:
        df (pd.DataFrame): pandas dataframe containing VI time-series data
        vi_column (str): the type of VI-index represented by time-series
        fill_method (str): method for interpolating NaN values in the middle of time-series
        resample_date (bool): whether or not to resample date into uniformity
        smoothing_window (int): Savitzky-Golay smoothing window size
        smoothing_polygon_order (int): Savitzky-Golay smoothing polygon order

    Returns:
        (pd.DataFrame): cleaned VI time-series dataset
    """
    df = df.copy() 

    drop_cols = ["system:index", ",geo"]
    if pd.Series(drop_cols).isin(df.columns).all():
        logger.info("Removing unnecessary columns...")
        df = df.drop(columns=drop_cols)

    if "uuid" in df.columns:
        # Re-order columns
        new_col_order = ["uuid"] + [col for col in df.columns if col != "uuid"]
        df = df.reindex(columns=new_col_order)
    else:
        raise KeyError("Column 'uuid' was not found in dataframe.")

    value_cols = df.columns.drop("uuid")

    if fill_method == "interpolate":
        df[value_cols] = df[value_cols].interpolate(method="linear", axis=1).bfill(axis=1).ffill(axis=1)
    elif fill_method == "linear":
        df[value_cols] = df[value_cols].apply(fill_dates, axis=1)
    else:
        raise ValueError(f"Unknown fill_method: {fill_method}")

    # Reshaping data from GEE native wide format
    df_melted = (
        df.melt(id_vars="uuid", var_name="date", value_name=vi_column)
          .assign(date=lambda x: pd.to_datetime(x["date"], format="%Y-%m-%d", errors="coerce"))
          .dropna(subset=["date"])
          .drop_duplicates(subset=["uuid", "date"], keep="first")
          .sort_values(["uuid", "date"])
          .reset_index(drop=True)
    )

    groups: list[pd.DataFrame] = []
    for uuid, group in df_melted.groupby("uuid", sort=False):
        group = group.copy()

        if len(group) >= smoothing_window:
            group[vi_column] = savgol_filter(group[vi_column], smoothing_window, smoothing_polygon_order)

        if resample_date:
            group = resample_vi_group(group, vi_column, uuid, resample_freq)

        groups.append(group)

    df_smoothed = pd.concat(groups, ignore_index=True)
    df_smoothed["outlier"] = (
        df_smoothed.groupby("uuid")[vi_column]
                   .transform(lambda s: find_outliers(s, n_estimators, outlier_contamination, random_state))
    )

    df_smoothed.loc[df_smoothed["outlier"] == -1, vi_column] = np.nan
    df_smoothed[vi_column] = (
        df_smoothed.groupby("uuid")[vi_column]
        .transform(lambda s: s.bfill().ffill())
    )

    df_clean = (
        df_smoothed.drop(columns="outlier")
        .assign(**{vi_column: lambda x: x[vi_column].clip(-1.0, 1.0)})
    )

    try:
        validator = VIDataValidator(vi_column)
        validator.validate(df_clean)

        logger.info("VI preprocessing completed successfully for index: %s", vi_column)

        return df_clean
    except pa.errors.SchemaError as e:
        logger.exception(f"Data validation failed: {e}")
        raise

def preprocess_vi_timeseries(
    raw_partitioned_data: dict[str, Callable[[], Any]], 
    params: dict[str, Any]
) -> dict[str, pd.DataFrame]:
    """
    Main entry-point for raw VI time-series data preprocessing pipeline
    """
    cleaned_vi_data_partitions: dict[str, pd.DataFrame] = {}

    vi_column = params["vi_column"]
    fill_method = params.get("fill_method", "interpolate")
    resample_date = params.get("resample_date", True)
    resample_freq = params.get("resample_freq", "5D")
    smoothing_window = params.get("smoothing_window", 7)
    smoothing_polygon_order = params.get("smoothing_polygon_order", 3)
    n_estimators = params.get("n_estimators", 150)
    outlier_contamination = params.get("outlier_contamination", 0.075)
    random_state = params.get("random_state", 10)

    for partition_idx, partition_load_func in raw_partitioned_data.items():
        raw_vi_data = partition_load_func()        

        cleaned_vi_data = clean_vi_series(
            df=raw_vi_data,
            vi_column=vi_column,
            fill_method=fill_method,
            resample_date=resample_date,
            resample_freq=resample_freq,
            smoothing_window=smoothing_window,
            smoothing_polygon_order=smoothing_polygon_order,
            n_estimators=n_estimators,
            outlier_contamination=outlier_contamination,
            random_state=random_state
        )
        
        cleaned_vi_data_partitions[partition_idx] = cleaned_vi_data
    
    return cleaned_vi_data_partitions