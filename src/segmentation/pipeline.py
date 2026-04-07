from __future__ import annotations

import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import geopandas as gpd

from src.common.logging_config import get_logger
from src.configs.segmentation import SegmentationConfig
from src.segmentation.s3_io import (
    TileJob,
    build_tile_jobs,
    discover_regions,
    download_file,
    make_s3_client,
    upload_file
)
from src.segmentation.sam_model import build_samgeo, cleanup_torch_cache

logger = get_logger(__name__)

@dataclass(slots=True)
class TileResult:
    """
    Class for storing results of processing job on a single GeoTIFF tile.

    Attributes:
        region (str): region name with trailing underscored number
        tile_name (str): tile name in the form of 'tile_xx'
        csv_path (Path): path to output csv file
        gpkg_path (Path): path to output gpkg file
        mask_tiff_path (Path): path to output mask tiff
    """
    region: str
    tile_name: str
    csv_path: Optional[Path] = None
    gpkg_path: Optional[Path] = None
    mask_tiff_path: Optional[Path] = None

def segment_single_tile(tile_job: TileJob, config: SegmentationConfig, sam) -> TileResult:
    region_work_dir = config.paths.scratch_dir / tile_job.region
    output_dir = region_work_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    mask_tiff_path = output_dir / f"{tile_job.tile_name}_delineated.tiff"
    gpkg_path = output_dir / f"{tile_job.tile_name}_delineated.gpkg"
    csv_path = output_dir / f"{tile_job.tile_name}_delineated.csv"

    sam.generate(
        str(tile_job.local_path),
        str(mask_tiff_path),
        batch=config.sam.batch,
        foreground=config.sam.foreground,
        erosion_kernel=config.sam.erosion_kernel,
        mask_multiplier=config.sam.mask_multiplier
    )

    gdf = gpd.read_file(gpkg_path)
    gdf["uuid"] = [uuid.uuid4() for _ in range(len(gdf))]
    df = gdf.to_wkt()
    df.to_csv(csv_path, index=False)

    if not config.io.emit_mask_tiff and mask_tiff_path.exists():
        mask_tiff_path.unlink()

    if not config.io.emit_gpkg and gpkg_path.exists():
        gpkg_path.unlink()

    if not config.io.emit_csv and csv_path.exists():
        csv_path.unlink()

    return TileResult(
        region=tile_job.tile_name,
        tile_name=tile_job.tile_name,
        csv_path=csv_path if config.io.emit_csv else None,
        gpkg_path=gpkg_path if config.io.emit_gpkg else None,
        mask_tiff_path=mask_tiff_path if config.io.emit_mask_tiff else None
    )