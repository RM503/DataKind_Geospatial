''' 
This script uses modules in imagery_generation.py and creates tiled images around
the different distributor locations
'''
import os 
import sys 
sys.path.insert(1, '/Users/rafidmahbub/Desktop/DataKind_Geospatial')
import config 
from google.cloud import storage 
import ee 
from s2cloudless import s2cloudless 
from imagery_generation import (
     generate_feature_collection,
     get_ndvi,
     get_evi,
     spectral_transform
)
import geopandas as gpd

''' 
Permissions and authentications
'''

gee_project = config.gee_project
ee.Authenticate()
ee.Initialize(project=gee_project)

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './../service_key_gcp.json'
storage_client = storage.Client()

def generate_image_tiles(covering_grid: ee.FeatureCollection, img_type='rgb') -> ee.ImageCollection:
    ''' 
    This function converts an RoI defined by a covering grid consisting of tiles of a particular size
    and converts them into a series of images in a collection.

    Args: (i) covering_grid - ee.FeatureCollection consisting of rectangles spanning the grid
          (ii) img_type -  argument for the type of image required; img_type can be `rgb`, `false`, `ndvi` and `evi`;
                           defaults to `rgb`
    Returns: ee.ImageCollection() object containing imagery contained within the tiles
    '''
    img_collection = [] # initialize an empty list to store images
    N = covering_grid.size().getInfo()

    for i in range(N):
        tile = covering_grid.getInfo()['features'][i]['geometry']['coordinates']
        tile_AOI = ee.Geometry.Polygon(tile)

        s2_sr_median = s2cloudless(tile_AOI)
        
        if img_type == 'rgb':
            band_list = ['B4', 'B3', 'B2'] # true color band list
            rgb_img = s2_sr_median.select(band_list)
            img_collection.append(rgb_img)

        elif img_type == 'false':
            band_list = ['B8', 'B4', 'B3'] # false color band list
            false_img = s2_sr_median.select(band_list)
            img_collection.append(s2_sr_median.select(false_img))

        elif img_type == 'ndvi':
            ndvi_img = get_ndvi(s2_sr_median).select('NDVI')
            img_collection.append(ndvi_img)

        elif img_type == 'hsv-NDVI':
            hsv_img = spectral_transform(s2_sr_median, 'NDVI')
            img_collection.append(hsv_img)

        elif img_type == 'hsv-EVI':
            hsv_img = spectral_transform(s2_sr_median, 'EVI')
            img_collection.append(hsv_img)

        else:
            raise ValueError(f'Unsupported image type: {img_type}')
        
    return ee.ImageCollection(img_collection)

def export_to_cloud(
        img_col: ee.ImageCollection,
        feature_col: ee.FeatureCollection,
        export_bucket: str,
        distributor_name: str,
        vis_params: dict,
        img_type: str='rgb'
) -> None:
    ''' 
    This function loops over the tiles in the covering grid of ONE LOCATION and exports
    the tiles as GeoTIFFs to a GCP storage bucket.

    Args: (i) img_col - ee.ImageCollection object containing the tiled images of one location
          (ii) feature_col - ee.FeatureCollection object storing polygon geometry of each tile
          (iii) export_bucket - name of GCP bucket to which images are exported (included subdirectories, if needed)
          (iv) distributor_name - name of distributor for identification
          (v) vis_params - a dictionary containing visualization parameters for image export
          (vi) img_type - type of image to be exported; defaults to `rgb`
    Returns: None
    '''
    N = img_col.size().getInfo()
    img_list = img_col.toList(N)

    for i in range(N):
        img = ee.Image(img_list.get(i))
        coords = feature_col.getInfo()['features'][i]['geometry']['coordinates']

        img_vis = img.visualize(**vis_params)
        '''
        Export parameters specification; images are generated in EPSG:4326 and saved
        as GeoTIFF files and a 10 px/m resolution.
        '''
        export_params = {
            'image': img_vis,
            'description': 'tile_' + str(i) + '_' + img_type,
            'bucket': export_bucket,
            'crs': 'EPSG:4326',
            'fileFormat': 'GeoTIFF',
            'scale': 10,
            'region': coords,
            'maxPixels' : 1e9
        }

        task = ee.batch.Export.image.toCloudStorage(**export_params)
        task.start()


gdf = gpd.read_file('./../geocoding/distributor_locations_priority_geocoded.json')

# collection of distributor locations and names 
feature_col = generate_feature_collection(gdf)
N = feature_col.size().getInfo()

for i in range(N):
    ''' 
    We loop over the the distributor locations, extracting their coordinates and generating 
    covering grids around them. The function `generated_image_tiles()` exports tiled images
    of the covering grid to cloud.
    '''
    dist_loc = ee.Geometry.Point(
        feature_col.getInfo()['features'][i]['geometry']['coordinates']
    )
    dist_name = feature_col.getInfo()['features'][i]['properties']['name']

    roi = dist_loc.buffer(1.5e4).bounds()
    grid = roi.coveringGrid('EPSG:4326', 5000)

    # Generate image tiles for location `i` for export to cloud
    img_tiles = generate_image_tiles(covering_grid=grid, img_type='hsv-EVI')
    vis_params = {'min': 0, 'max': 1.0}
    
    export_to_cloud(
        img_col=img_tiles,
        feature_col=grid,
        export_bucket='s2_image_tiles/' + dist_name + '/',
        distributor_name=dist_name, 
        vis_params=vis_params,
        img_type='hsv-EVI'
    )
