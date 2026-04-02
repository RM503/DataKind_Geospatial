# Utility functions for generating raster images
from __future__ import annotations

import logging
import os

import cv2
import ee
import numpy as np
import geopandas as gpd
from sentinelhub import BBox, SHConfig, geo_utils
from shapely import make_valid
from shapely.geometry import MultiPolygon, Polygon
from tqdm import tqdm

from common.logging_config import get_logger

logger = get_logger(__name__)

# GEE configurations
GEE_PROJECT = os.getenv("GEE_PROJECT")
ee.Authenticate()
ee.Initialize()

# SentinelHub configurations
config = SHConfig(os.getenv("SENTINELHUB_USER"))

def generate_covering_grid(
        gdf: gpd.GeoDataFrame,
        *,
        crs: str = "EPSG:4326",
        buffer_distance: float=10_000,
        scale: int=5000
) -> gpd.GeoDataFrame:
    """
    This function takes a geopandas dataframe and creates a covering grid of the AoI
    buffer zone.

    Args:
        gdf (geopandas.GeoDataFrame): the geopandas dataframe
        crs (str): the CRS of the geometry; defaults to EPSG:4326
        buffer_distance (float): the buffer distance in meters, defaults to 10_000
        scale (int): nominal grid scale passed to Earth Engine's coveringGrid

    Returns:
        (geopandas.GeoDataFrame): a copy of the geopandas dataframe with the a new
        'tile_grids' column containing MultiPolygon geometries
    """

    gdf = gdf.copy()

    if gdf.crs is None:
        logging.info(f"No CRS found for geodataframe; setting CRS to {crs}.")
        gdf = gdf.to_crs(crs)

    # Check for invalid geometries
    if (~gdf.geometry.is_valid()).any():
        logging.info(f"Invalid geometry(ies) found; repairing them.")
        gdf.geometry = gdf.geometry.apply(make_valid)

    gdf.tile_grids = None

    for idx, row in tqdm(gdf.iterrows(), total=len(gdf)):
        lon, lat = row["Longitude"], row["Latitude"]
        center = ee.Geometry.Point([lon, lat])
        aoi = center.buffer(buffer_distance).bounds()
        covering_grid = aoi.coveringGrid(proj=crs, scale=scale)

        # Retrieve features
        grid_info = covering_grid.getInfo()
        features = grid_info.get("features", [])

        polygons = []
        for feature in features:
            coords = feature["geometry"]["coordinates"]

            # Ensure that coveringGrid produces polygons
            if feature["geometry"]["type"] == "Polygon":
                polygons.append(Polygon(coords[0]))
            elif feature["geometry"]["type"] == "MultiPolygon":
                for poly_coords in coords:
                    polygons.append(Polygon(poly_coords[0]))

        gdf.at[idx, "tile_grids"] = MultiPolygon(polygons) if polygons else MultiPolygon()

    return gdf

def generate_lon_lat(aoi_bbox: BBox, aoi_size: tuple, resolution: int) -> tuple[np.ndarray, np.ndarray]:
    """
    This function generates longitude and latitude axes from bounding box and resolution
    information to be used in the generation of raster tiles.

    Args:
          aoi_bbox (BBox): the bounding box of the AoI
          aoi_size (tuple): size of the bounding box
          resolution (int): the resolution of the bounding box

    Returns:
          (tuple[np.ndarray, np.ndarray]): a tuple containing the longitude and latitude axes
    """
    # First convert the bbox to UTM
    bbox_utm = geo_utils.to_utm_bbox(aoi_bbox)
    transform = bbox_utm.get_transform_vector(resx=resolution, resy=resolution)

    pix_lon = np.array(np.arange(0, aoi_size[0]))
    lons = np.array([pix_lon] * aoi_size[1])

    pix_lat = np.array(np.arange(0, aoi_size[1]))
    lats = np.array([pix_lat] * aoi_size[0]).transpose()

    lon, lat = geo_utils.pixel_to_utm(lats, lons, transform)

    lon_degrees, lat_degrees = geo_utils.transform_point(lon, lat, bbox_utm.crs)

    return lon_degrees[0,:], lat_degrees[:,0]

def edge_enhance(img: np.ndarray, kernel_size: int=11, wf: int=2) -> np.ndarray:
    """
    This function performs edge enhancement on input images by superposing a 
    Gaussian blur subtracted version on top of the image.

    Args:
          img (np.ndarray): input image as numpy array
          kernel_size (int): pixel area over which Gaussian blur is applied; defaults to 11 px
          wf (int): a weight factor; defaults to 2

    Returns: enhanced image as numpy array
    """
    img_blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    img_sharpened = img + wf * (img - img_blurred)
    return img_sharpened