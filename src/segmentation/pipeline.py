from __future__ import annotations

import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import geopandas as gpd
from mypy_boto3_s3 import S3Client

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
    """
    Runs SAM model on a single tile, producing artifacts.

    Args:
        tile_job (TileJob): the TileJob dataclass
        config (SegmentationConfig): full segmentation configuration dataclass

    Returns:
        (TileResult): dataclass containing artifact information after segmentation
    """
    region_work_dir = config.paths.scratch_dir / tile_job.region
    output_dir = region_work_dir / "outputs" # Output analog of scratch_dir/region/inputs
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
    sam.tiff_to_gpkg(str(mask_tiff_path), str(gpkg_path), simplify_tolerance=None)

    # Create dataframes with uuid for each delineation
    gdf = gpd.read_file(gpkg_path)
    gdf["uuid"] = [uuid.uuid4() for _ in range(len(gdf))]
    df = gdf.to_wkt()
    df.to_csv(csv_path, index=False)

    # Clean up paths that are no longer required
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

def upload_tile_outputs(s3_client: S3Client, config: SegmentationConfig, result: TileResult) -> None:
    """Uploads a tile output"""
    region_prefix = f"{config.io.output_prefix.rstrip("/")}/{result.region}"
    
    if result.csv_path and result.csv_path.exists():
        upload_file(
            s3_client=s3_client,
            local_path=result.csv_path,
            bucket=config.output_bucket,
            key=f"{region_prefix}/csv/{result.csv_path.name}"
        )

    if result.gpkg_path and result.gpkg_path.exists():
        upload_file(
            s3_client=s3_client,
            local_path=result.gpkg_path,
            bucket=config.output_bucket,
            key=f"{region_prefix}/gpkg/{result.gpkg_path.name}",
        )

    if result.mask_tiff_path and result.mask_tiff_path.exists():
        upload_file(
            s3_client=s3_client,
            local_path=result.mask_tiff_path,
            bucket=config.output_bucket,
            key=f"{region_prefix}/mask_tiffs/{result.mask_tiff_path.name}",
        )

def run_region(config: SegmentationConfig, region: str) -> None:
    """Runs segmentation job on all tiles in a given region."""
    s3_client = make_s3_client(region_name=config.aws.aws_region_name)

    # Generate all metadata for region
    tile_jobs = build_tile_jobs(
        s3_client=s3_client,
        bucket_name=config.input_bucket,
        region=region,
        scratch_root=config.paths.scratch_dir,
        suffix=config.io.input_suffix
    )

    if not tile_jobs:
        logger.warning(f"No tiles found for region {region}")
        return 

    sam_kwargs = config.sam.load_sam_kwargs()
    
    sam = build_samgeo(
        checkpoint_path=config.model.checkpoint_path,
        sam_kwargs=sam_kwargs,
        model_type=config.model.model_type,
        device=config.model.device
    )

    for idx, tile_job in enumerate(tile_jobs):
        logger.info(f"Region={region}, tile={idx}, key={tile_job.s3_key}")

        # download tiles to scratch_dir/region/inputs
        download_file(
            s3_client=s3_client,
            bucket=config.input_bucket,
            key=tile_job.s3_key,
            destination=tile_job.local_path
        )

        result = segment_single_tile(tile_job=tile_job, config=config, sam=sam)
        upload_tile_outputs(s3_client=s3_client, config=config, result=result)
        cleanup_torch_cache()

        if config.runtime.cleanup_local_files:
            try:
                tile_job.local_path.unlike(missing_ok=True)
            except TypeError:
                if tile_job.local_path.exists():
                    tile_job.local_path.unlink()

    # Global cleanup
    if config.runtime.cleanup_local_files:
        region_root = config.paths.scratch_dir / region 
        if region_root.exists():
            shutil.rmtree(region_root, ignore_errors=True)

def run_all_regions(config: SegmentationConfig) -> list[str]:
    """Runs segmentation on all regions."""
    s3_client = make_s3_client(region_name=config.aws.aws_region_name)
    discovered_regions = discover_regions(
        s3_client=s3_client,
        bucket_name=config.input_bucket,
        suffix=config.io.input_suffix
    )

    if config.region_allowlist:
        allowset = set(config.region_allowlist)
        regions = [region for region in discovered_regions if region in allowset]
    else:
        regions = discovered_regions 

    for region in regions:
        logger.info(f"Currently processessing region: {region}")
        run_region(config=config, region=region)

    return regions