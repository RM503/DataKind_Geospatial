'''
Utility functions
'''
import time 
import sys 
sys.path.insert(1, '/Users/rafidmahbub/Desktop/DataKind_Geospatial')
import config
import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
from shapely import bounds, make_valid
from shapely.geometry import Polygon, MultiPolygon
import ee

from sentinelhub import (
    SHConfig,
    DataCollection,
    SentinelHubCatalog,
    SentinelHubRequest,
    BBox,
    bbox_to_dimensions,
    CRS,
    MimeType
)
import cv2 
from typing import Generator

# GEE configurations
gee_project = config.gee_project
ee.Authenticate()
ee.Initialize(project=gee_project)

# SentinelHub configurations
config = SHConfig('rmahbub503')

def get_covering_grid(
        gdf: gpd.GeoDataFrame,
        buffer_size: float=1e4,
        scale: int=5000
) -> gpd.GeoDataFrame:
    ''' 
    This function takes in a geopandas dataframe and creates a covering grid of the
    AoI buffer zone.

    Args: (i) gdf - priority distributor location gdf
          (ii) buffer_size - size of the buffer zone in sq. meters; defaults to 10000 sq. meters
          (iii) scale - width of each tiles; defaults to 5000 meters

    Returns: modified gdf with tile_grids column
    '''
    gdf = gdf.copy()

    # Setting default CRS where absent
    if gdf.crs is None:
        gdf = gdf.set_crs('EPSG:4326')

    # Fix invalid geometries (if present)
    if (gdf.geometry.is_valid==False).any():
        gdf.geometry = make_valid(gdf.geometry)

    # Initializing tile_grid column
    gdf['tile_grids'] = None
    
    # For iterating through the different distributor locations
    for idx, row in tqdm(gdf.iterrows()):
        lon = float(row['Longitude'])
        lat = float(row['Latitude'])

        center = ee.Geometry.Point([lon, lat])
        AoI = center.buffer(buffer_size).bounds()
        covering_grid = AoI.coveringGrid('EPSG:4326', scale=scale)

        N = covering_grid.size().getInfo()
        polygons = []
        
        # For iterating through each tile in covering grid
        for i in range(N):
            tile = covering_grid.getInfo()['features'][i]['geometry']['coordinates']
            polygon = Polygon(tile[0])
            polygons.append(polygon)

        multipolygon = MultiPolygon(polygons)
        gdf.at[idx, 'tile_grids'] = multipolygon
        #row['tile_grids'] = multipolygon

    #gdf.to_file('tiles.csv', driver='GPKG')
    return gdf

def SH_request_builder(
        tiles: MultiPolygon,
        start_date: str,
        end_date: str,
        resolution: int=5,
        img_type: str='true_color_optimized',
) -> Generator[np.ndarray, None, None]:
    '''
    This function acts a generator that yields image tiles stored in MultiPolygon geometries.
    It uses SentinelHub's request builder and user defined evalscripts for iamge generation.

    Args: (i) tiles - covering grid tiles of a distributor location buffer zone as a MultiPolygon geometry
          (ii) start_date
          (iii) end_date
          (iv) resolution - pixel resolution of the images; defaults to 5 px/m
          (v) img_type - type of image to generate where the choice picks out a particular evalscript; defaults to `true_color_optimized`

    Yields: image tiles as numpy arrays
    '''
    for idx, tile in enumerate(tiles.geoms):
        AoI = tile
        xmin, ymin, xmax, ymax = bounds(AoI) # extract corner information of AoI
        AoI_bbox = BBox(
            [xmin, ymin, xmax, ymax],
            CRS.WGS84
        )

        AoI_size = bbox_to_dimensions(AoI_bbox, resolution=resolution)

        catalog = SentinelHubCatalog(config=config)
        time_interval = start_date, end_date 
        search_iterator = catalog.search(
            DataCollection.SENTINEL2_L2A,
            bbox=AoI_bbox,
            time=time_interval,
            fields={"include": ["id", "properties.datetime"], "exclude": []},
        )   

        search_results = list(search_iterator)
        
        # Read evalscript for request builder
        with open('evalscripts/' + img_type + '.js', 'r') as f:
            evalscript = f.read()

        ''' 
        In `mosaickingOrder`, the `leastCC` implies image files with the lowest percentage
        of cloudy pixels. This is automatically handled using s2cloudless.
        '''
        request = SentinelHubRequest(
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
            responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],
            bbox=AoI_bbox,
            size=AoI_size,
            config=config,
        )

        r = request.get_data()
        img = r[0]
        yield img

        # Yields image and sleeps for 0.1 s to prevent Request error
        time.sleep(0.1)

def edge_enhance(img: np.ndarray, kernel_size: int=11, wf: int=2) -> np.ndarray:
    ''' 
    This function performs edge enhancement on input images by superposing a 
    Gaussian blur subtracted version on top of the image.

    Args: (i) img - input image as numpy array
          (ii) kernel_size - pixel area over which Gaussian blur is applied; defaults to 11 px
          (iii) wf - a weight factor; defaults to 2

    Returns: enhanced image as numpy array
    '''
    img_blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    img_sharpened = img + wf * (img - img_blurred)
    return img_sharpened
