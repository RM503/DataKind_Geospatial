# Raster generation configuration

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

PersistType = Literal["upload_to_s3", "save_locally"]

@dataclass(frozen=True)
class RasterGenerationConfig:
    """
    Configuration class for raster generation.

    Attributes:
        # Raster generation
        start_date (str): start date of raster; defaults to 2024/01/01
        end_date (str): end date of raster; defaults to 2025/01/1
        resolution (int): resolution of raster; defaults to 5 px/m
        img_type (str): type of raster; defaults to 'TIFF'
        evalscript_type (str): type of raster; defaults to 'hightlight_optimized'
        buffer_distance (float): distance in meters to buffer raster; defaults to 10_000
        grid_scale (int): scale of raster; defaults to 5_000
        geometry_crs (str): CRS of raster; defaults to EPSG:4326

        # local export
        output_dir (Optional[Path]): directory to save rasters; defaults to None

        # s3 export
        bucket_name (Optional[Path]): S3 bucket name; defaults to None
        s3_prefix (str): S3 prefix; defaults to ''

        # evalscript directory
        evalscript_dir (Optional[Path]): directory to fetch evalscripts; defaults to None
    """
    # Raster generation
    start_date: str = "2024/01/01"
    end_date: str = "2025/01/01"
    resolution: int = 5
    img_type: str = "TIFF"
    evalscript_type: str = "highlight_optimized"
    buffer_distance: float = 10_000
    grid_scale: int = 5_000
    geometry_crs: str = "EPSG:4326"

    # local export
    output_dir: Optional[Path] = None

    # s3 export
    bucket_name: Optional[str] = None
    s3_prefix: str = ""

    # evalscript directory
    evalscript_dir: Optional[Path] = None

    def resolved_evalscript_dir(self) -> Path:
        if self.evalscript_dir is not None:
            return Path(self.evalscript_dir)

        return Path(__file__).resolve().parent / "evalscripts"

def sentinelhub_config_name() -> Optional[str]:
    """Returns SentinelHub config profile name, if available."""
    return os.getenv("SETINELHUB_USER")

def get_gee_project() -> Optional[str]:
    """Returns GEE project name, if available."""
    return os.getenv("GEE_PROJECT")