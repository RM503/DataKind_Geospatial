# I/O functionalities for GeoTIFF retrieval and saving in S3 buckets

from __future__ import annotations

from dataclasses import astuple, dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Optional

import boto3
import numpy as np
import rasterio as rio

from common.logging_config import get_logger

logger = get_logger(__name__)

@dataclass(frozen=True)
class RasterTile:
    """
    Holds raster tile information to be used in I/O operations

    Attributes:
        img (np.ndarray): the tile array of shape (H, W, C)
        lats (np.ndarray): the latitudes of the tile
        lons (np.ndarray): the longitudes of the tile
    """
    img: np.ndarray
    lats: np.ndarray
    lons: np.ndarray

    def __post_init__(self):
        if self.img.ndim != 3:
            raise ValueError(f"Image array must have 3 dimensions (H, W, C); got {self.img.ndim}")

@dataclass
class GeoTiffWriter:
    """
    Holds stable export config for a data generation pipeline. It can be
    instantiated once and `save()` or `upload()` methods called per tile.

    Attributes:
        crs (str): the CRS of the image; defaults to EPSG:4326
        s3_client (boto3.client('s3')): the S3 client object
        bucket_name (str): the S3 bucket name for file persistence; defaults to None
        s3_prefix (str): the S3 prefix for file persistence (named per region)
        output_dir (Path): directory for local persistence; defaults to None
    """
    crs: str = "EPSG:4326"

    # S3 config
    s3_client: object = field(default=None, repr=False)
    bucket_name: Optional[str] = None
    s3_prefix: str = ""

    # Local save config
    output_dir: Optional[Path] = None

    def __post_init__(self):
        # Post initialization
        if self.s3_client is None and self.bucket_name is not None:
            self.s3_client = boto3.client("s3")
        if self.output_dir is not None:
            # Coerce output_dir to be Path object
            self.output_dir = Path(self.output_dir)

    def to_buffer(self, tile: RasterTile) -> BytesIO:
        """
        Converts tile arrays into in-memory GeoTIFF buffer.

        Args:
            tile (RasterTile): the raster tile array

        Returns:
            (BytesIO): the GeoTIFF buffer stream
        """
        img, lats, lons = astuple(tile)

        # Find image corners
        left, right = np.min(lons), np.max(lons)
        bottom, top = np.min(lats), np.max(lats)

        img_3bands = np.stack((img[:, :, i] for i in range(img.shape[2])))

        transform = rio.transform.from_bounds(
            left, bottom, right, top,
            img.shape[1], img.shape[0]
        )

        metadata = {
            "driver": "GTiff",
            "height": img.shape[0],
            "width": img.shape[1],
            "count": img.shape[2],
            "dtype": img.dtype,
            "crs": self.crs,
            "transform": transform
        }

        # Create an in-memory buffer
        buffer = BytesIO()
        with rio.open(buffer, "w", **metadata) as dst:
            dst.write(img_3bands)

        buffer.seek(0)

        return buffer

    def save(self, tile: RasterTile, filename: str) -> Path:
        """
        Writes an image array to a GeoTIFF file on disk.

        Args:
            tile (RasterTile): the raster tile array
            filename (str): the filename to which the image will be saved

        Returns:
            (Path): the path to the GeoTIFF file
        """
        if self.output_dir is None:
            raise ValueError("output_dir must be set to save locally.")

        img, lats, lons = astuple(tile)

        output_path = self.output_dir / filename
        output_path.mkdir(parents=True, exist_ok=True)

        buffer = self.to_buffer(img, lats, lons, self.crs)
        output_path.write_bytes(buffer.read())

        return output_path

    def upload(self, tile: RasterTile, filename: str) -> str:
        """
        Writes an image array to a GeoTIFF file on disk.

        Args:
            tile (RasterTile): the raster tile array
            filename (str): the filename to which the image will be saved

        Returns:
            (str): the s3_key
        """
        if self.bucket_name is None:
            raise ValueError("bucket_name must be set to upload to S3.")

        s3_key = f"{self.s3_prefix}/{filename}".lstrip("/")
        self.s3_client.upload_fileobj(
            self.to_buffer(tile),
            self.bucket_name,
            s3_key,
            ExtraArgs={"ContentType": "image/tiff"}
        )

        return s3_key

    def export(self, tile: RasterTile, filename: str) -> None:
        if self.output_dir is not None:
            path = self.save(tile, filename)
            logger.info(f"Saving {filename} to {path}")

        if self.bucket_name is not None:
            key = self.upload(tile, filename)
            logger.info(f"Uploaded {filename} to {key}")