"""
Modules for extracting and merging NDVI, NDMI and polygon data using Dask
for memory efficiency.
"""

from __future__ import annotations

import logging

import dask.dataframe as dd
import geopandas as gpd
import pandas as pd
from scipy.signal import savgol_filter

logger = logging.getLogger(__name__)

def _perform_checks(ddf: dd.DataFrame):
    if not isinstance(ddf, dd.DataFrame):
        logger.warning("Input is not a Dask DataFrame. This may lead to memory issues.")
    if any(col.startswith("Unnamed") for col in ddf.columns):
        logger.info("Dropping columns starting with 'Unnamed'...")
        ddf = ddf.loc[:, ~ddf.columns.str.startswith("Unnamed")]
    return ddf


def _smooth_partition(
        df: pd.DataFrame,
        window_size: int = 7,
        polygon_order: int = 3
    ) -> pd.DataFrame:
    def apply_savgol(group: pd.DataFrame) -> pd.DataFrame:
        if len(group) >= window_size:
            group = group.sort_values("date")
            group["ndvi"] = savgol_filter(
                group["ndvi"],
                window_length=window_size,
                polyorder=polygon_order
            )
            group["ndmi"] = savgol_filter(
                group["ndmi"],
                window_length=window_size,
                polyorder=polygon_order
            )
        return group
    return df.groupby("uuid", group_keys=False).apply(apply_savgol)


def extract_merge_vi_data(
        ddf_ndvi: dd.DataFrame,
        ddf_ndmi: dd.DataFrame,
        gdf: gpd.GeoDataFrame,
        npartitions: int = 20,
        smooth: bool = True,
        smoothing_kwargs: dict | None = None
) -> pd.DataFrame:
    """
    Merges cleaned NDVI and NDMI data using Dask for memory efficiency.

    Args:
        ddf_ndvi (dd.DataFrame): Cleaned NDVI time-series data
        ddf_ndmi (dd.DataFrame): Cleaned NDMI time-series data
        gdf (gpd.GeoDataFrame): Polygons with 'uuid' and 'prediction_decoded'
        smooth (bool): Whether to apply Savitzky-Golay smoothing
        npartitions (int): Number of desired partitions

    Returns:
        pd.DataFrame: Merged and optionally smoothed NDVI/NDMI time-series data
    """
    ddf_ndvi = _perform_checks(ddf_ndvi)
    ddf_ndmi = _perform_checks(ddf_ndmi)

    # Create a mapping of uuid to prediction_decoded
    gdf_unique = gdf.drop_duplicates(subset="uuid")
    uuid_list = gdf_unique["uuid"].tolist()
    polygon_mapping = dict(zip(gdf_unique["uuid"], gdf_unique["prediction_decoded"]))

    ddf_ndvi = ddf_ndvi[ddf_ndvi.uuid.isin(uuid_list)]
    ddf_ndmi = ddf_ndmi[ddf_ndmi.uuid.isin(uuid_list)]

    ddf_merged = dd.merge(
        ddf_ndvi,
        ddf_ndmi,
        on=["uuid", "date"],
        how="inner"
    )

    df = pd.DataFrame({"uuid": uuid_list})
    df["partition_id"] = pd.qcut(
        df.index,
        q=min(len(df), npartitions),
        labels=False
    )
    partition_mapping = dict(zip(df["uuid"], df["partition_id"]))

    # Map partition_id to each uuid
    ddf_merged["partition_id"] = (
        ddf_merged["uuid"].map(
            partition_mapping,
            meta=("partition_id", "int64")
        )
    )

    ddf_merged = (
        ddf_merged
        .set_index("partition_id", shuffle="tasks")
        .reset_index(drop=True)
        .map_partitions(lambda df: df.sort_values(["uuid", "date"]))
        .assign(
            polygon_type=lambda x: x["uuid"].map(polygon_mapping, meta=("polygon_type", "object"))
        )
    )

    if smooth:
        if smoothing_kwargs is None:
            smoothing_kwargs = {}
        ddf_merged = ddf_merged.map_partitions(
            _smooth_partition,
            **smoothing_kwargs
        )

    return ddf_merged.compute()
