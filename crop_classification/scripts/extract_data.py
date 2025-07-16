# Merge NDVI and NDMI time-series data in a memory-safe way
import geopandas as gpd
import pandas as pd
import dask.dataframe as dd
import logging
from scipy.signal import savgol_filter

def smooth_partition(df):
    def apply_savgol(group):
        if len(group) >= 7:
            group = group.sort_values("date")
            group["ndvi"] = savgol_filter(group["ndvi"], window_length=7, polyorder=3)
            group["ndmi"] = savgol_filter(group["ndmi"], window_length=7, polyorder=3)
        else:
            group["ndvi"] = group["ndvi"]
            group["ndmi"] = group["ndmi"]
        return group

    return df.groupby("uuid", group_keys=False).apply(apply_savgol)

def extract_data(
    ddf_ndvi: dd.DataFrame,
    ddf_ndmi: dd.DataFrame,
    gdf: gpd.GeoDataFrame,
    smooth: bool = True,
    npartitions: int = 20,
) -> pd.DataFrame:
    """
    Merges cleaned NDVI and NDMI data using Dask for memory efficiency.

    Args:
        ddf_ndvi (dd.DataFrame): Cleaned NDVI data
        ddf_ndmi (dd.DataFrame): Cleaned NDMI data
        gdf (gpd.GeoDataFrame): Polygons with 'uuid' and 'prediction_decoded'
        smooth (bool): Whether to apply Savitzky-Golay smoothing
        npartitions (int): Number of desired partitions

    Returns:
        pd.DataFrame: Merged and optionally smoothed NDVI/NDMI data
    """
    # Type checks
    if not isinstance(ddf_ndvi, dd.DataFrame) or not isinstance(ddf_ndmi, dd.DataFrame):
        logging.warning("Either ddf_ndvi or ddf_ndmi is not a Dask DataFrame.")
    if not isinstance(gdf, gpd.GeoDataFrame):
        logging.warning("gdf is not a GeoDataFrame.")

    # Prepare uuid list and polygon type mapping (safe deduplication)
    gdf_unique = gdf.drop_duplicates(subset="uuid")
    uuid_list = gdf_unique["uuid"].tolist()
    polygon_mapping = dict(zip(gdf_unique["uuid"], gdf_unique["prediction_decoded"]))

    # Filter NDVI/NDMI data by uuid and drop `Unnamed` columns
    ddf_ndvi = ddf_ndvi[ddf_ndvi.uuid.isin(uuid_list)]
    ddf_ndmi = ddf_ndmi[ddf_ndmi.uuid.isin(uuid_list)]

    ddf_ndvi = ddf_ndvi.loc[:, ~ddf_ndvi.columns.str.startswith("Unnamed")]
    ddf_ndmi = ddf_ndmi.loc[:, ~ddf_ndmi.columns.str.startswith("Unnamed")]

    ddf_merged = dd.merge(ddf_ndvi, ddf_ndmi, on=["uuid", "date"], how="inner")

    """
    We must be careful while partitioning since time-series data per uuid
    must not be split between partitions and remain contiguous. Hence, we
    bin the uuid into npartition partitions.
    """
    uuid_df = pd.DataFrame({"uuid": uuid_list})
    uuid_df["partition_id"] = pd.qcut(uuid_df.index, q=min(len(uuid_df), npartitions), labels=False)
    uuid_to_partition = dict(zip(uuid_df["uuid"], uuid_df["partition_id"]))

    # Map partition_id to each uuid
    ddf_merged["partition_id"] = ddf_merged["uuid"].map(uuid_to_partition, meta=("partition_id", "int64"))

    # Shuffle by partition_id to group similar uuid together and sort each partition by uuid and date
    ddf_merged = (
        ddf_merged.set_index("partition_id", shuffle="tasks")
        .reset_index(drop=True)
        .map_partitions(lambda df: df.sort_values(["uuid", "date"]))
    )

    # Add polygon type column
    ddf_merged["polygon_type"] = ddf_merged["uuid"].map(polygon_mapping, meta=("polygon_type", "object"))

    # Optional smoothing
    if smooth:
        ddf_merged = ddf_merged.map_partitions(smooth_partition)

    # Don't forget to apply .compute() to final result
    return ddf_merged.compute()