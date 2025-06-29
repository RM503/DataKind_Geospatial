""" 
This file contains functions for generating vegetation index time-series data from delineated polygons using
capabilities of Google Earth Engine.
"""
import os 
import dotenv
import geopandas as gpd 
import ee
import json
import shapely
from shapely.geometry import Polygon  
from datetime import datetime
import logging 

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("vi_series.log"),
        logging.StreamHandler()
    ]
)

dotenv.load_dotenv()
GEE_PROJECT = os.getenv("GEE_PROJECT")

ee.Authenticate()
ee.Initialize(project=GEE_PROJECT)

class VIIndex:
    """
    This class packages functions for calculating various vegetation indices. 
    The various functions are packaged as static methods.
    """
    @staticmethod
    def add_NDVI(img: ee.Image) -> ee.Image:
        """ 
        This function takes an ee.Image object and adds an NDVI band to it.

        Args: img - ee.Image object

        Returns: same ee.Image object with an NDVI band 
        """
        ndvi = img.normalizedDifference(["B8", "B4"]).rename("ndvi")
        return img.addBands([ndvi])
    
    @staticmethod
    def add_NDMI(img: ee.Image) -> ee.Image:
        """ 
        This function takes an ee.Image object and adds an NDWI band to it. This uses
        the following NDWI convention

        NDMI = (NIR - SWIR) / (NIR + SWIR)

        Args: img - ee.Image object

        Returns: same ee.Image object with an NDWI band 
        """
        ndmi = img.normalizedDifference(["B8", "B11"]).rename("ndmi")
        return img.addBands([ndmi])
    
    @staticmethod
    def add_EVI(img: ee.Image) -> ee.Image:
        """ 
        This function takes an ee.Image object and adds an EVI band to it.

        Args: img - ee.Image object

        Returns: same ee.Image object with an EVI band 
        """

        # EVI calculations are sensitive to scaling
        evi = img.expression(
            "2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))", {
                "NIR": img.select("B8").divide(10000),
                "RED": img.select("B4").divide(10000),
                "BLUE": img.select("B2").divide(10000)
            }
        ).rename("evi")
        return img.addBands([evi])

def return_ee_geometry(poly: Polygon) -> ee.Geometry:

    """ 
    This function acts on a shapely.geometry polygon object and 
    converts it to an ee.Geometry object.

    Args: poly - shapely.geometry object

    Returns: ee.Geometry object
    """
    if not poly.is_valid:
        logging.info("Making polygon valid")
        poly = shapely.make_valid(poly)
    # shapely.to_geojson() returns a string object which needs to be
    # converted to geojson before using with ee.Geometry
    RoI = ee.Geometry(json.loads(shapely.to_geojson(poly)))

    return RoI

def gdf_to_FeatureCollection(gdf: gpd.GeoDataFrame) -> ee.FeatureCollection:
    """ 
    This function converts each RoI in the geodataframe into an
    ee.Feature object with attached metadata. All the features are
    then aggregated into an ee.FeatureCollection object.
    """
    features = []

    # Iterates through each row in the geodataframe and attaches metadata
    # Extermely important to retrieve information regarding polygons
    for _, row in gdf.iterrows():
        feature = ee.Feature(row["RoI"])
        metadata = {
            "region": row["region"],
            "tile_name": row["tile_name"],
            "uuid": row["uuid"]
        }
        feature = feature.set(metadata)

        features.append(feature)
    
    return ee.FeatureCollection(features)

def mask_cloud_and_shadow(img: ee.Image) -> ee.Image:
    """ 
    This function creates a pixel mask that are deemed to be cloud
    and (or) cloud shadow using Sentinel-2 `MSK_CLDPRB` and
    `Scene Classification Layer`.

    Args: img - ee.Image object

    Returns: same ee.Image object with an updated mask
    """

    # The amount of cloud probability will affect number of sample points
    cloud_prob = img.select("MSK_CLDPRB")
    snow_prob = img.select("MSK_SNWPRB")
    cloud = cloud_prob.lt(30)
    snow = snow_prob.lt(30)

    # Use SCL to select shadows and cirrus cloud masks
    scl = img.select("SCL")
    shadow = scl.eq(3)
    cirrus = scl.eq(10)

    mask = cloud.And(snow).And(cirrus.neq(1)).And(shadow.neq(1))

    return img.updateMask(mask)

def make_IndexCollection(
        start_date: str,
        end_date: str,
        FeatureCollection: ee.FeatureCollection,
        index_type: str
) -> ee.ImageCollection:
    """ 
    This function returns an image collection between in the required date range
    for all the geometries in an ee.FeatureCollection object, returning a calculated
    NDVI band.
    
    Args: (i) start_date - start date of image collection
          (ii) end_date - end date of image collection
          (iii) FeatureCollection - ee.FeatureCollection object containing polygon information

    Returns: ee.ImageCollection object
    """ 
    if datetime.strptime(start_date, "%Y-%m-%d") < datetime(2016, 6, 13):
        logging.warning("COPERNICUS S2_SR_HARMONIZED images do not extend beyond this date.")

    if not isinstance(FeatureCollection, ee.FeatureCollection):
        logging.error("FeatureCollection must be of type ee.FeatureCollection")

    index_type = index_type.lower() # for consistency

    index_funcs = {
        "ndvi": (VIIndex.add_NDVI, "ndvi"),
        "ndmi": (VIIndex.add_NDMI, "ndmi"),
        "evi": (VIIndex.add_EVI, "evi")
    }

    if index_type not in index_funcs:
        logging.error(f"Unsupported index type: {index_type}")

    add_index_func, index_band = index_funcs[index_type]

    img_collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterDate(start_date, end_date)
          .filterBounds(FeatureCollection).map(mask_cloud_and_shadow).map(add_index_func)
    ).select(index_band) # Create the image collection

    def map_index(img: ee.Image):
        """ 
        This function applies a reducer to each image in a collection and
        performs a spatial mean of the NDVI.
        """
        stats = img.reduceRegions(
            collection=FeatureCollection,
            reducer=ee.Reducer.mean().setOutputs([index_band]),
            scale=10
        ).filter(ee.Filter.neq(index_band, None)) # Filter out null NDVI values

        def set_date(feature: ee.Feature):
            return feature.set("date", img.date().format("YYYY-MM-dd"))
        
        return stats.map(set_date)
    
    return img_collection.map(map_index).flatten()

def format_table(
        table: ee.ImageCollection,
        row_id: str, 
        col_id: str,
        index_type: str
) -> ee.FeatureCollection:
    """ 
    This function creates a table for storing time-series results for each polygon in a tile
    in a row-major order.

    Args: (i) table - ee.ImageCollection object containing NDVI results
          (ii) row_id - row elements (ideally uuid)
          (iii) col_id - column elements (ideally date)

    Returns: ee.FeatureCollection object 
    """
    # Get unique rows based on rowId
    rows = table.distinct(row_id)

    # Create join condition
    condition = ee.Filter.equals(leftField=row_id, rightField=row_id)

    # Perform the join
    joined = ee.Join.saveAll('matches').apply(primary=rows, secondary=table, condition=condition)

    # Map over joined results
    def map_row(row):
        matches = ee.List(row.get('matches'))

        def extract_values(feature):
            feature = ee.Feature(feature)
            ndvi_val = ee.List([feature.get(index_type), -9999]).reduce(ee.Reducer.firstNonNull())
            return [feature.get(col_id), ee.Number(ndvi_val).format('%.3f')]

        # Get list of [colId, ndvi] pairs
        values = matches.map(extract_values)

        # Flatten and convert to dictionary
        flat_values = ee.Dictionary(ee.List(values).flatten())

        # Return row with the wide-format values added
        return row.select([row_id]).set(flat_values)

    return ee.FeatureCollection(joined.map(map_row))

def main(index_type: str="ndvi") -> None:
    """ 
    Main function for performing all previous tasks and uploading calculation results to Google Drive.
    """
    # Regions with good satellite images
    regions = ["Kajiado_1", "Kajiado_2", "Laikipia_1", "Machakos_1", "Mashuru_1"]

    tiles = [f"tile_{i}" for i in range(25)]

    GDF_DIR = "../../../samgeo_aws_ec2/vectors/"

    # Iterate over regions for export
    for region in regions:

        # Read polygon geometry data and create an RoI column
        gdf = (
            gpd.read_file(f"{GDF_DIR}{region}.gpkg")
               .to_crs(epsg=4326)
               .assign(
                   RoI = lambda x: x.geometry.apply(return_ee_geometry)
               ) 
        )
        logging.info(f"Calculating {index_type} for {region}")

        for tile in tiles:
            logging.info(f"Processing {tile} for {region}")

            gdf_tile = gdf[gdf["tile_name"] == tile]
            geometries = gdf_to_FeatureCollection(gdf_tile)

            START_DATE = "2020-01-01"
            END_DATE = "2024-12-31" 

            img_collection = make_IndexCollection(START_DATE, END_DATE, geometries, index_type)
            index_table = format_table(img_collection, "uuid", "date", index_type)

            # Preparing for export
            table_name = f"{index_type}_series_{region}_{tile}"
            
            task = ee.batch.Export.table.toDrive(
                collection=index_table,
                description=table_name,
                folder=f"{index_type}_series",
                fileFormat="CSV"
            )

            task.start()

if __name__ == "__main__":
    main(index_type="ndmi")