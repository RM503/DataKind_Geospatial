"""
CLI entry-point for Earth Engine modules. Runs tasks and exports VI timeseries
data to Google Drive.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import ee
import geopandas as gpd

from .geometry import prepare_gdf, gdf_to_feature_collection
from .io import load_assets_config, resolve_geometry_paths
from .vegetation_indices import VegetationIndexSpec, get_vegetation_index
from .vi_timeseries import format_table, make_index_collection
from src.common.logging_config import get_logger

logger = get_logger(__name__)


def run_ee_task(
        gdf: gpd.GeoDataFrame,
        vi_index_name: str,
        start_date: str,
        end_date: str,
        scale: int = 10
) -> None:
    """
    Starts Earth Engine table export tasks for vegetation-index time series.

    Loads Earth Engine asset configuration, groups the input geometries by tile,
    computes the requested vegetation index for each tile, pivots the resulting
    time series into export-ready tables, and submits one CSV export task per
    tile to Google Drive.

    Args:
        (i) gdf (gpd.GeoDataFrame): Input geometries and metadata prepared for Earth
            Engine processing.
        (ii) vi_index_name (str): Registered vegetation index name to compute.
        (iii) start_date (str): Start date for the Sentinel-2 query in `YYYY-MM-DD`
            format.
        (iv) end_date (str): End date for the Sentinel-2 query in `YYYY-MM-DD`
            format.
        (v) scale (int): Earth Engine reduction scale in meters. Defaults to 10.
    """
    config = load_assets_config()
    data_columns = config["geometry_assets"].get("data_columns", {})
    uuid_col = data_columns.get("uuid", "uuid")
    tile_col = data_columns.get("tile_name", "tile_name")

    if tile_col not in gdf.columns:
        raise ValueError(f"Expected tile name column '{tile_col}' not found in GeoDataFrame.")

    tiles = gdf[tile_col].unique()

    index_spec: VegetationIndexSpec = get_vegetation_index(vi_index_name)
    index_spec.scale = scale # Overwrite default scale with user-provided value

    for tile in tiles:
        logger.info(f"Processing tile: {tile}")
        gdf_tile = gdf[gdf[tile_col] == tile]
        geometries = gdf_to_feature_collection(gdf_tile)

        # Create the index image collection and format it for export
        img_collection = make_index_collection(start_date, end_date, geometries, index_spec)
        index_table = format_table(img_collection, uuid_col, "date", index_spec.band_name)
        # Prepare export task
        table_name = f"{index_spec.name}_series_{tile}"

        task = ee.batch.Export.table.toDrive(
            collection=index_table,
            description=table_name,
            folder=f"{index_spec.name}_series",
            fileFormat="CSV"
        )
        task.start()
        logger.info(f"Started export task for tile {tile} with name: {table_name}")


def build_parser() -> argparse.ArgumentParser:
    """
    Builds the command-line argument parser for Earth Engine processing.

    Returns:
        (argparse.ArgumentParser): Parser configured with vegetation-index, date
        range, and scale options.
    """
    parser = argparse.ArgumentParser(description="Run Earth Engine processing.")

    parser.add_argument("--vi_index_name", required=True)
    parser.add_argument("--vi_index_band", required=True)
    parser.add_argument("--scale", type=int, default=10)

    parser.add_argument("--start_date", required=True)
    parser.add_argument("--end_date", required=True)

    return parser


def main() -> None:
    """
    Runs the Earth Engine CLI workflow.

    Parses command-line arguments, resolves configured geometry asset paths,
    prepares each geometry file, and launches Earth Engine export tasks for the
    requested vegetation-index time series.
    """
    parser = build_parser()
    args = parser.parse_args()

    # Retrieve geometry paths from config
    geometry_paths: list[Path] = resolve_geometry_paths()
    for path in geometry_paths:
        logger.info(f"Processing geometries for region: {path.stem}")

        gdf = prepare_gdf(path)
        logger.info(f"Loaded {len(gdf)} geometries from {path}")

        run_ee_task(
            gdf=gdf,
            vi_index_name=args.vi_index_name,
            start_date=args.start_date,
            end_date=args.end_date,
            scale=args.scale
        )


if __name__ == "__main__":
    main()
