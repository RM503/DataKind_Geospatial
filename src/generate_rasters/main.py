from __future__ import annotations

from pathlib import Path
from typing import Optional

import geopandas as gpd
from tqdm import tqdm

from configs.raster_generation import RasterGenerationConfig
from .geometry import has_empty_tile_grids, generate_covering_grid, initialize_ee
from .io import GeoTiffWriter
from .naming import append_running_count, slugify
from .request_builder import fetch_tile, iter_requests
from common.logging_config import get_logger

logger = get_logger(__name__)

def build_writer(config: RasterGenerationConfig, subdir_name: Optional[str]=None) -> GeoTiffWriter:
    """
    Builds a GeoTiffWriter object for a given config.

    Args:
        config (RasterGenerationConfig): the configuration for raster generation.
        subdir_name (str, optional): the subdirectory to use for the output GeoTiffWriter (either local or S3);
                                     defaults to None.

    Returns:
        GeoTiffWriter object
    """
    # if local persistence is required
    output_dir = None
    if config.output_dir is not None:
        output_dir = Path(config.output_dir)
        if subdir_name:
            output_dir = Path(output_dir) / subdir_name

    # if S3 upload is required
    s3_prefix = config.s3_prefix.strip("/")
    if subdir_name:
        s3_prefix = f"{s3_prefix}/{subdir_name}".strip("/")

    return GeoTiffWriter(
        crs=config.geometry_crs,
        output_dir=output_dir,
        bucket_name=config.bucket_name,
        s3_prefix=s3_prefix,
    )

def validate_input_gdf(
    gdf: gpd.GeoDataFrame,
    county_col: str = "County",
    lon_col: str = "Longitude",
    lat_col: str = "Latitude",
) -> None:
    """
    Validates that required columns exist.
    """
    required_cols = {county_col, lon_col, lat_col, "geometry"}
    missing = required_cols.difference(gdf.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

def generate_rasters_for_gdf(
    gdf: gpd.GeoDataFrame,
    config: RasterGenerationConfig,
    county_col: str = "County",
    lon_col: str = "Longitude",
    lat_col: str = "Latitude",
) -> None:
    """
    Main raster generation pipeline for a GeoDataFrame of locations.

    Args:
        gdf (gpd.GeoDataFrame): the geodataframe containing locations for raster generation
        config (RasterGenerationConfig): the configuration for raster generation.
        county_col (str, optional): the name of the county column in the gdf. Defaults to "County".
        lon_col (str, optional): the name of the longitude column in the gdf. Defaults to "Longitude".
        lat_col (str, optional): the name of the latitude column in the gdf. Defaults to "Latitude".

    Returns:
        None
    """
    validate_input_gdf(gdf, county_col=county_col, lon_col=lon_col, lat_col=lat_col)

    initialize_ee()

    gdf_w_grids = generate_covering_grid(
        gdf,
        crs=config.geometry_crs,
        buffer_distance=config.buffer_distance,
        scale=config.grid_scale,
        lon_col=lon_col,
        lat_col=lat_col,
    )

    if has_empty_tile_grids(gdf_w_grids, col_name="tile_grids"):
        msg = "Either 'tile_grids' is missing, null, or one or more entries are empty."
        logger.error(msg)
        raise ValueError(msg)

    gdf_w_grids = append_running_count(
        gdf_w_grids,
        col_name=county_col,
        new_col_name=f"{county_col}_enumerated",
    )

    evalscript_dir = config.resolved_evalscript_dir()

    for _, row in tqdm(
        gdf_w_grids.iterrows(),
        total=len(gdf_w_grids),
        desc="Generating rasters",
    ):
        region_name = row[f"{county_col}_enumerated"]
        subdir_name = slugify(region_name)

        logger.info(f"Processing location: {region_name}")

        writer = build_writer(config=config, subdir_name=subdir_name)

        requests = iter_requests(
            tiles=row["tile_grids"],
            start_date=config.start_date,
            end_date=config.end_date,
            evalscript_dir=evalscript_dir,
            evalscript_type=config.evalscript_type,
            resolution=config.resolution,
            data_folder=None,
        )

        for tile_idx, (request, aoi_bbox, aoi_size) in enumerate(requests):
            tile = fetch_tile(
                request=request,
                aoi_bbox=aoi_bbox,
                aoi_size=aoi_size,
                resolution=config.resolution,
            )

            filename = f"tile_{tile_idx}.tiff"
            writer.export(tile=tile, filename=filename)

            logger.info(f"Exported {filename} for {region_name}")