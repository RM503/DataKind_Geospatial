# SentinelHub request builder

import time
from pathlib import Path
from typing import Generator, Optional

import numpy as np
from sentinelhub import (
    BBox,
    CRS,
    DataCollection,
    MimeType,
    SHConfig,
    SentinelHubRequest,
    bbox_to_dimensions,
)
from shapely.geometry import MultiPolygon, Polygon

from .config import get_sentinelhub_config_name
from .geometry import generate_lon_lat
from .io import RasterTile
from common.logging_config import get_logger

logger = get_logger(__name__)

IterRequestsReturn = Generator[tuple[SentinelHubRequest, BBox, tuple[int, int]], None, None]

def get_sh_config() -> SHConfig:
    """Builds SentinelHub config."""
    profile_name = get_sentinelhub_config_name()
    if profile_name:
        return SHConfig(profile_name=profile_name)
    return SHConfig()

def read_evalscript(evalscript_dir: Path, evalscript_type: str) -> str:
    # Read evalscript from file
    script_path = Path(evalscript_dir) / f"{evalscript_type}.js"
    if not script_path.exists():
        raise FileNotFoundError(f"Evalscript not found: {script_path}")

    return script_path.read_text(encoding="utf-8")

def build_single_request(
    tile: Polygon,
    start_date: str,
    end_date: str,
    evalscript_dir: Path,
    evalscript_type: str,
    resolution: int=5,
    data_folder: Optional[Path] = None
) -> tuple[SentinelHubRequest, BBox, tuple[int, int]]:
    """
    Builds a SentinelHubRequest object for a single tile geometry.

    Args:
        tile (Polygon): the tile to build request for
        start_date (str): the start date of the request
        end_date (str): the end date of the request
        evalscript_dir (Path): the path to the evalscript file
        evalscript_type (str): the type of evalscript to use
        resolution (int): the resolution to use; defaults to 5 px/m
        data_folder (Optional[Path]): the directory to export the requests to
    """
    # SentinelHub request body
    evalscript = read_evalscript(evalscript_dir=evalscript_dir, evalscript_type=evalscript_type)

    xmin, ymin, xmax, ymax = tile.bounds
    aoi_bbox = BBox([xmin, ymin, xmax, ymax], CRS.WGS84)
    aoi_size = bbox_to_dimensions(aoi_bbox, resolution=resolution)
    
    """
    In `mosaickingOrder`, the `leastCC` implies image files with the lowest percentage
    of cloudy pixels. This is automatically handled using s2cloudless.
    """
    request = SentinelHubRequest(
        data_folder=data_folder,
        evalscript=evalscript,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A.define_from(
                    name="s2l2a", service_url="https://sh.dataspace.copernicus.eu"
                ),
                time_interval=(start_date, end_date),
                other_args={"dataFilter": {"mosaickingOrder": "leastCC"}},
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=aoi_bbox,
        size=aoi_size,
        config=get_sh_config(),
    )

    return request, aoi_bbox, aoi_size

def iter_requests(
    tiles: MultiPolygon,
    start_date: str,
    end_date: str,
    evalscript_dir: Path,
    evalscript_type: str,
    resolution: int=5,
    data_folder: Optional[Path] = None,
) -> IterRequestsReturn:
    """
    Iterates over `build_single_request` to generate GeoTIFFs for all polygons
    that are required.

    Args:
        tiles (MultiPolygon): the tiles to build requests for
        start_date (str): the start date of the request
        end_date (str): the end date of the request
        evalscript_dir (Path): the path to the evalscript file
        evalscript_type (str): the type of evalscript to use
        resolution (int): the resolution to use; defaults to 5 px/m
        data_folder (Optional[Path]): the directory to export the requests to

    Yields:
        image tiles as SentinelHubRequest objects
    """

    for tile in tiles.geoms:
        yield build_single_request(
            tile=tile,
            start_date=start_date,
            end_date=end_date,
            evalscript_dir=evalscript_dir,
            evalscript_type=evalscript_type,
            resolution=resolution,
            data_folder=data_folder
        )

        time.sleep(0.1)

def fetch_tile(
    request: SentinelHubRequest,
    aoi_bbox: BBox,
    aoi_size: tuple[int, int],
    resolution: int,
) -> RasterTile:
    """Retrieves a raster tile from the SentinelHubRequest object."""
    response = request.get_data()

    if not response:
        raise RuntimeError("SentinelHub request returned no data.")

    img = response[0]

    if img.ndim == 2:
        img = np.expand_dims(img, axis=-1)

    lons, lats = generate_lon_lat(aoi_bbox, aoi_size, resolution)

    return RasterTile(img=img, lats=lats, lons=lons)