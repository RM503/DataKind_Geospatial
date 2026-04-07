from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import boto3
from botocore.exceptions import ClientError
from mypy_boto3_s3 import S3Client

from common.logging_config import get_logger

logger = get_logger(__name__)

@dataclass(slots=True)
class TileJob:
    region: str
    s3_key: str
    local_path: Path

    @property
    def tile_name(self) -> str:
        return self.local_path.stem

def make_s3_client(region_name: Optional[str]=None) -> S3Client:
    try:
        s3_client = boto3.client("s3", region_name=region_name)
        return s3_client
    except ClientError as e:
        raise e

def list_all_keys(
        s3_client: S3Client,
        bucket_name: str,
        prefix: str
) -> list[str]:
    paginator = s3_client.get_paginator("list_objects_v2")
    keys: list[str] = []

    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]

            if not key.endswith("/"):
                keys.append(key)

    return keys

def discover_regions(s3_client: S3Client, bucket_name: str) -> list[str]:
    response = s3_client.list_objects_v2(Bucket=bucket_name)
    regions: set[str] = set()

    if "Contents" in response:
        for obj in response.get("Contents", []):
            key = obj["Key"]

            if key.endswith("/"):
                region = key.rstrip("/")
                regions.add(region)

    return sorted(regions)

def list_region_tile_keys(
        s3_client: S3Client,
        bucket_name: str,
        region: str,
        suffix: str = ".tiff"
) -> list[str]:
    keys = list_all_keys(s3_client, bucket_name, region)

    return sorted([key for key in keys if key.lower().endswith(suffix.lower())])

def build_tile_jobs(
        s3_client: S3Client,
        bucket_name: str,
        region: str,
        scratch_root: Path,
        suffix: str = ".tiff"
) -> list[TileJob]:
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
        destination: Path
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading s3://{bucket_name}{key} -> {destination}")
    s3_client.download_file(bucket_name, key, str(destination))

    return destination

def upload_file(
        s3_client: S3Client,
        local_path: Path,
        bucket_name: str,
        key: str
) -> None:
    logger.info(f"Uploading {local_path} -> s3://{bucket_name}{key}")
    s3_client.upload_file(str(local_path), bucket_name, key)

def object_exists(
        s3_client: S3Client,
        bucket_name: str,
        key: str,
) -> bool:
    try:
        s3_client.head_object(Bucket=bucket_name, Key=key)
        return True
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code")
        if error_code in {"404", "NoSuchKey","NotFound"}:
            return False
        raise