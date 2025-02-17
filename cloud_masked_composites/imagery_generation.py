'''
This is a Python script for generating Sentinel-2 imagery around the distributor locations. The code is the
same as imagery_generation.ipynb.

The `s2cloudless` module contains an implementation of the s2cloudless method for performing cloud-masking.
'''

import geopandas as gpd
import ee 
from s2cloudless import s2cloudless

ee.Authenticate()
ee.Initialize(project='project503-430216')

def generate_feature_collection(gdf: gpd.GeoDataFrame) -> ee.FeatureCollection:
    ''' 
    This function generates an ee.FeatureCollection() object. It takes in the geopandas dataframe
    as input and extracts longitude, latitude information along with distributor names. Then,
    these are converted to ee.Feature() objects and finally returned as a collection.

    Args: geopandas dataframe with longitude and latitude information
    Returns: ee.FeatureCollection() object with distributor coordinates to be used as the center of region's buffer
    '''
    # Extract longitude, latitude and distributors from dataframe
    longitudes = gdf['Longitude'].tolist()
    latitudes = gdf['Latitude'].tolist()
    distributors = gdf['Distributors'].tolist()

    # Create a nested list of coordinates and clean the distributor names
    coord_list = [[lon, lat] for lon, lat in zip(longitudes, latitudes)]
    distributors_list = [distributor.lower().replace(' ', '_') for distributor in distributors]

    point_collection = []

    for i in range(len(longitudes)):
        coords = coord_list[i]
        point = ee.Geometry.Point(coords)
        feature = ee.Feature(point, {'name': distributors_list[i]})
        point_collection.append(feature)

    return ee.FeatureCollection(point_collection) 

def get_ndvi(img: ee.Image) -> ee.Image:
    ''' 
    This function computes the NDVI of an image and adds it as a new band.

    Args: ee.Image() object
    Returns: same ee.Image() object with an additional `NDVI` band
    '''
    ndvi = img.normalizedDifference(['B8', 'B4']).rename('NDVI')
    return img.addBands(ndvi)

def get_evi(img: ee.Image) -> ee.Image:
    ''' 
    This function computes the EVI of an image and adds it as a new band.

    Args: ee.Image() object
    Returns: same ee.Image() object with an additional `NDVI` band
    '''
    evi = img.expression(
        '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))', {
            'NIR': img.select('B8'),
            'RED': img.select('B4'),
            'BLUE': img.select('B2')
        }
    ).rename('EVI')
    return img.addBands(evi)

def spectral_transform(img: ee.Image, hsv_v: str) -> ee.Image:
    ''' 
    This function takes in an RGB image as input and returns a spectral transformed counterpart. It does so
    by transforming the RGB image into HSV space, where HSV stands for `Hue`, `Saturation` and `Value` (in this
    case, `Value` is either NDVI or EVI). 
    '''
    rgb_img = img.select(['B4', 'B3', 'B2'])

    if hsv_v == 'NDVI':
        rgb_img_v = get_ndvi(img)

    elif hsv_v == 'EVI':
        rgb_img_v = get_evi(img)

    else:
        raise ValueError(f'Unsupported value type: {hsv_v}')

    hsv_img = rgb_img.rgbToHsv()

    transformed_img = ee.Image.cat(
        [
            hsv_img.select('hue'),
            hsv_img.select('saturation'),
            rgb_img_v.select(hsv_v)
        ]
    ).hsvToRgb()

    return transformed_img

def generate_image_collection(
        feature_collection: ee.FeatureCollection,
        img_type: str = 'rgb',
        buffer: float = 1.5e4
    ) -> ee.ImageCollection:
    ''' 
    This function generates an ee.ImageCollection() object. It takes in the list of coordinates 
    as a ee.FeatureCollection() object. It then iterates over each feature in the collection,
    applying the s2cloudless function and generating the median composite. Then, the list of all
    images is converted to an ee.ImageCollection() object.

    Args: (i) ee.FeatureCollection() object with all the distributor location coordinates
          (ii) img_type -  argument for the type of image required; img_type can be `rgb`, `false`, `ndvi` and `hsv`;
                           defaults to `rgb`
          (iii) buffer - size of the buffer zone around the distributor location in meters; defaults to 15000 m (15 km)
    Returns: ee.ImageCollection() object containing images within the buffer around each location at the required band(s)
    '''
    img_collection = []
    N = feature_collection.size().getInfo() # image_collection size
    for i in range(N):
        coords = feature_collection.getInfo()['features'][i]['geometry']['coordinates']
        AOI = ee.Geometry.Point(coords).buffer(buffer).bounds()
        # Generate cloud-masked median composites
        s2_sr_median = s2cloudless(AOI)

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

def export_img_col(
        img_col: ee.ImageCollection,
        feature_col: ee.FeatureCollection,
        export_folder: str,
        vis_params: dict,
        img_type: str = 'rgb',
        buffer: float = 1.5e4
) -> None:
    ''' 
    This function takes in an image collection, feature collection and other parameters and
    exports all the images in the collection iteratively. The images are exported as GeoTIFF
    files in the WGS 84 format.

    Args: (i) img_col - ee.ImageCollection() object containing the images to be exported
          (ii) feature_col - ee.FeatureCollection() object containing distributor locations and
                             names as additional data
          (iii) export_folder - name of Google Drive folder where images are to be exported
          (iV) vis_params - a dictionary containing visualization parameters for image export
          (v) img_type - type of image to be exported; defaults to `rgb`
          (vi) buffer - size of buffer for clipping; defaults to 15 km
    Returns: None
    '''
    # Convert the image collection into a list of images
    N = img_col.size().getInfo()
    img_list = img_col.toList(N)

    for i in range(N):
        img = ee.Image(img_list.get(i))

        # Get name of distributor contained in feature collection
        dist_name = feature_col.getInfo()['features'][i]['properties']['name']
        coords = feature_col.getInfo()['features'][i]['geometry']['coordinates']
        AOI = ee.Geometry.Point(coords).buffer(buffer).bounds()

        img_vis = img.visualize(**vis_params)

        '''
        Export parameters specification; images are generated in EPSG:4326 and saved
        as GeoTIFF files and a 10 px/m resolution.
        '''
        export_params = {
                'image': img_vis,
                'description': dist_name + '_' + img_type,
                'folder': export_folder,
                'crs': 'EPSG:4326',
                'fileFormat': 'GeoTIFF',
                'scale': 10,
                'region': AOI.getInfo()['coordinates'],
                'maxPixels' : 1e9
            }

        task = ee.batch.Export.image.toDrive(**export_params)
        task.start()

if __name__ == '__main___':
	gdf = gpd.read_file('./../geocoding/distributor_locations_priority_geocoded.json')

	# collection of distributor locations and names 
	feature_col = generate_feature_collection(gdf)

	hsv_img_col = generate_image_collection(feature_col, img_type='hsv')

	vis_params = {'min': 0, 'max': 1.0}

	export_img_col(
	    hsv_img_col,
	    feature_col,
	    'imagery',
	    vis_params,
	    img_type='hsv-EVI'
	)