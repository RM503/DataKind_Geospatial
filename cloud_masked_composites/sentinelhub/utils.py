'''
Utility functions
'''
import dotenv
import logging
import time 
import config
import numpy as np
import geopandas as gpd
from tqdm import tqdm
from shapely import bounds, make_valid
from shapely.geometry import Polygon, MultiPolygon
import ee

from sentinelhub import (
    SHConfig,
    DataCollection,
    SentinelHubRequest,
    BBox,
    bbox_to_dimensions,
    CRS,
    MimeType,
    geo_utils
)
import cv2 
from typing import Union, Generator
import rasterio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("ndvi_series.log"),
        logging.StreamHandler()
    ]
)

dotenv.load_dotenv()
GEE_PROJECT = os.getenv("GEE_PROJECT")

# GEE configurations
gee_project = config.gee_project
ee.Authenticate()
ee.Initialize(project=gee_project)

# SentinelHub configurations
config = SHConfig('rmahbub503')

def generate_covering_grid(
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
        logging.info('CRS information missing from gdf; setting it to EPSG: 4326')
        gdf = gdf.set_crs('EPSG:4326')

    # Fix invalid geometries (if present)
    if (gdf.geometry.is_valid==False).any():
        logging.info('Invalid geometries present; fixing them')
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

    return gdf

def generate_lon_lat(
        AoI_bbox: BBox,
        AoI_size: tuple,
        resolution: int 
) -> tuple[np.ndarray, np.ndarray]:
    ''' 
    This function generates longitude and latitude axes from bounding box and resolution
    information to be used in the generation of raster tiles.

    Args: (i) AoI_bbox - coordinates of the corners of bounding box
          (ii) AoI_size - pixel size of the bounding box
          (iii) resolution - resolution of the image in px/m

     Returns: a tuple of longitude and latitude axes     
    '''
    # First convert the bbox to UTM
    bbox_utm = geo_utils.to_utm_bbox(AoI_bbox)
    transform = bbox_utm.get_transform_vector(resx=resolution, resy=resolution)

    pix_lon = np.array(np.arange(0, AoI_size[0]))
    lons = np.array([pix_lon]*AoI_size[1])

    pix_lat = np.array(np.arange(0, AoI_size[1]))
    lats = np.array([pix_lat] * AoI_size[0]).transpose()

    lon, lat = geo_utils.pixel_to_utm(lats, lons, transform)

    lon_degrees, lat_degrees = geo_utils.to_wgs84(lon, lat, bbox_utm.crs)

    return lon_degrees[0,:], lat_degrees[:,0]

def array_to_geotiff(
        img: np.ndarray,
        output_path: str,
        lats: np.ndarray,
        lons: np.ndarray
) -> None:
    '''
    This function writes the image arrays into geotiff files.
    '''

    # Find image corners
    left = np.min(lons)
    right = np.max(lons)
    bottom = np.min(lats)
    top = np.max(lats)

    # x_res = (right - left)/img.shape[1]
    # y_res = (top - bottom)/img.shape[0]

    band1 = img[:, :,0]
    band2 = img[:, :,1]
    band3 = img[:, :,2]

    img_3bands = np.stack((band1, band2, band3))

    transform = rasterio.transform.from_bounds(left, bottom, right, top, img.shape[1], img.shape[0])

    metadata = {
        'driver': 'GTiff',
        'height': img.shape[0],
        'width': img.shape[1],
        'count': 3,
        'dtype': img.dtype,
        'crs': 'EPSG:4326',
        'transform': transform
    }

    with rasterio.open(output_path, 'w', **metadata) as dst:
        dst.write(img_3bands)


def SH_request_builder(
        tiles: MultiPolygon,
        start_date: str,
        end_date: str,
        export_dir: str,
        evalscript_type: str,
        img_type: str,
        resolution: int=5,
) -> Generator[Union[np.ndarray, SentinelHubRequest], None, None]:
    '''
    This function acts a generator that yields image tiles stored in MultiPolygon geometries.
    It uses SentinelHub's request builder and user defined evalscripts for iamge generation.

    Args: (i) tiles - covering grid tiles of a distributor location buffer zone as a MultiPolygon geometry
          (ii) start_date
          (iii) end_date
          (iv) evalscript - type of image to generate where the choice picks out a particular evalscript
          (v) img_type - type of image to produce; supported types are 'PNG' or 'TIFF'
          (vi) resolution - pixel resolution of the images; defaults to 5 px/m

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
        
        # Read evalscript for request builder
        # The evalscript read context-manager depends on the `evalscript` parameter
        with open('evalscripts/' + evalscript_type + '.js', 'r') as f:
            evalscript = f.read()

        # Store the mimetypes here
        mimetypes = {
            'PNG': MimeType.PNG,
            'TIFF': MimeType.TIFF
        }
        
        mimetype = mimetypes[img_type]

        ''' 
        In `mosaickingOrder`, the `leastCC` implies image files with the lowest percentage
        of cloudy pixels. This is automatically handled using s2cloudless.
        '''
        request = SentinelHubRequest(
            data_folder = export_dir,
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
            responses=[SentinelHubRequest.output_response("default", mimetype)],
            bbox=AoI_bbox,
            size=AoI_size,
            config=config,
        )

        ''' 
        Subsequent processing will depend on mimetype. 
        '''
        if mimetype == MimeType.PNG:
            # PNG images that require more processing will be yielded with longitude and latitude information
            img = request.get_data()[0]
            lon_array, lat_array = generate_lon_lat(
                AoI_bbox=AoI_bbox,
                AoI_size=AoI_size,
                resolution=resolution
            )
            yield img, lon_array, lat_array

            # Yields image and sleeps for 0.1 s to prevent Request error
            time.sleep(0.1)

        elif mimetype == MimeType.TIFF:
            # GeoTIFF images that only require export
            yield request

            time.sleep(0.1)

        else:
            logging.warning('Workflow may not support this mimetype.')

            yield img

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