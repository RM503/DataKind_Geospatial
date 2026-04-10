"""
This module contains I/O functionalities for segmentation jobs on SageMaker.
In S3 input bucket, the GeoTIFFs are stored in the following manner:

Region_A_i --> tile_0.tiff, tile_1.tiff, ..., tile_24.tiff
Region_B_i --> tile_0.tiff, tile_1.tiff, ..., tile_24.tiff
...

where subscript i is an integer representing multiple buffers of the same county.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import boto3
from botocore.exceptions import ClientError
from mypy_boto3_s3 import S3Client

from src.common.logging_config import get_logger

logger = get_logger(__name__)

@dataclass(slots=True)
class TileJob:
    """Stores attributes for segmentation job on a single tile"""
    region: str
    s3_key: str
    local_path: Path

    @property
    def tile_name(self) -> str:
        return self.local_path.stem

def make_s3_client(region_name: Optional[str] = None) -> S3Client:
    """Returns S3 client"""
    return boto3.client("s3", region_name=region_name)

def list_all_keys(
    s3_client: S3Client,
    bucket_name: str,
    prefix: str,
) -> list[str]:
    """
    Return all object keys under a prefix (excluding directory markers).
    """
    paginator = s3_client.get_paginator("list_objects_v2")
    keys: list[str] = []

    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]

            if not key.endswith("/"):
                keys.append(key)

    return keys

def discover_regions(s3_client: S3Client, bucket_name: str) -> list[str]:
    """Discover region "folders" at the bucket root."""
    regions: set[str] = set()

    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket_name, Delimiter="/"):
        for prefix in page.get("CommonPrefixes", []):
            region = prefix.get("Prefix")
            if region:
                regions.add(region.rstrip("/"))

    return sorted(regions)

def list_region_tile_keys(
    s3_client: S3Client,
    bucket_name: str,
    region: str,
    suffix: str = ".tiff",
) -> list[str]:
    """Only returns those keys that correspond to GeoTIFF files"""
    prefix = region if region.endswith("/") else f"{region}/"
    keys = list_all_keys(s3_client, bucket_name, prefix)

    return sorted([key for key in keys if key.lower().endswith(suffix.lower())])

def build_tile_jobs(
    s3_client: S3Client,
    bucket_name: str,
    region: str,
    scratch_root: Path,
    suffix: str = ".tiff",
) -> list[TileJob]:
    """
    This function builds job metadata for all GeoTIFF tiles present for a given region
    that is used by the segmentation pipeline.

    Args:
        s3_client (S3Client): the boto3 client object for S3 I/O operations
        bucket_name (str): the input bucket name for retrieving GeoTIFF tiles
        region (str): the region name for GeoTIFF tiles
        scratch_root (Path): the temporary directory where GeoTIFF inputs are to be kept
        suffix (str): file extension

    Returns:
        list[TileJob]: a list of job metadata in the `TileJob` format
    """
    region_dir = scratch_root / region / "inputs"
    region_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[TileJob] = []
    for key in list_region_tile_keys(s3_client, bucket_name, region, suffix):
        filename = Path(key).name
        jobs.append(
            TileJob(
                region=region,
                s3_key=key,
                local_path=region_dir / filename
            )
        )

    return jobs

def download_file(
    s3_client: S3Client,
    bucket_name: str,
    key: str,
    destination: Path,
) -> Path:
    """
    Download a single S3 object to disk.

    `destination` must be a concrete file path, or a directory path (no file
    suffix): in the latter case the object is written as
    `destination / basename(key)`. Boto3 cannot download directly onto a
    directory path.
    """
    if destination.is_dir():
        dest = destination / Path(key).name
    elif not destination.suffix and not destination.is_file():
        dest = destination / Path(key).name
    else:
        dest = destination

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading s3://{bucket_name}/{key} -> {dest}")

    try:
        s3_client.download_file(bucket_name, key, str(dest))
    except ClientError as e:
        raise e

    return dest

def upload_file(
    s3_client: S3Client,
    local_path: Path,
    bucket_name: str,
    key: str,
) -> None:
    """Uploads segmentation results from scratch directory to output S3 bucket"""
    logger.info(f"Uploading {local_path} -> s3://{bucket_name}/{key}")
    s3_client.upload_file(str(local_path), bucket_name, key)

def object_exists(
    s3_client: S3Client,
    bucket_name: str,
    key: str,
) -> bool:
    """Checks if object exists for the given key."""
    try:
        s3_client.head_object(Bucket=bucket_name, Key=key)
        return True
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code")
        if error_code in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise