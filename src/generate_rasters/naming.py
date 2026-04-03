from __future__ import annotations

import re

import geopandas as gpd

def append_running_count(gdf: gpd.GeoDataFrame, col_name: str, new_col_name: str) -> gpd.GeoDataFrame:
    """
    Adds a 1-based running count within each `col_name` group and creates a
    new enumerated name column.

    Example:
        County A, County A, County B, County C, ...
        -->
        County A 1, County A 2, County B, County C, ...

    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame containing county data
        col_name (str): Name of column to append count
        new_col_name (str): Name of new column

    Returns:
        gpd.GeoDataFrame: GeoDataFrame containing county data
    """
    gdf = gdf.copy()
    gdf["running_count"] = gdf.groupby(col_name).cumcount() + 1
    gdf[new_col_name] = gdf.apply(lambda row: f"{row[col_name]} {row['running_count']}", axis=1)

    return gdf.drop(columns=["running_count"])

def slugify(value: str) -> str:
    """Converts a string to a filesystem-friend slug."""
    value = value.strip().replace("/", "-")
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"[^A-Za-z0-9_\-]]", "", value)
    return value