"""
Script for generating MODIS and CHIRPS data on gridded region
"""
import os 
import dotenv
import ee
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon
from vi_time_series import return_ee_geometry, format_table
import logging 

logging.basicConfig(level=logging.INFO)

dotenv.load_dotenv()
GEE_PROJECT = os.getenv("GEE_PROJECT")

ee.Initialize()

def generate_pixel_grid(gdf, region: str, scale=1000) -> gpd.GeoDataFrame:
    """ 
    This function tiles the total geometric bounds of a geopandas dataframe.

    Args: (i) gdf - the farm polygon geodataframe for extracting bounds
          (ii) region - the region of interest
          (iii) scale - resolution of dataset (defaults to 1000 meters)

    Returns: grid - tiled grid of the total bounds of the geodataframe
    """
    # First convert crs to EPSG: 3857 to accurately tile total_bounds

    if gdf.crs != "EPSG:3857":
        gdf_proj = gdf.to_crs("EPSG:3857")

    bounds = gdf_proj.total_bounds
    xmin, ymin, xmax, ymax = bounds

    cols = list(np.arange(xmin, xmax + scale, scale))
    rows = list(np.arange(ymin, ymax + scale, scale))

    # Initialize a list to store polygons
    polygons = []

    for x in cols[:-1]:
        for y in rows[:-1]:
            polygons.append(Polygon([(x, y), (x + scale, y), (x + scale, y + scale), (x, y + scale)]))
    
    # Create grid GeoDataFrame and reproject back to WGS84
    grid = gpd.GeoDataFrame(geometry=polygons, crs="EPSG:3857").to_crs("EPSG:4326")
    grid["region"] = region 
    grid["tile_name"] = grid.index

    return grid[["region", "tile_name", "geometry"]]

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
            "tile_name": row["tile_name"]
        }
        feature = feature.set(metadata)

        features.append(feature)
    
    return ee.FeatureCollection(features)

def generate_MODIS_timeseries(
        FeatureCollection: ee.FeatureCollection,
        START_DATE: str="2000-01-01",
        END_DATE: str="2024-12-31",
        index_type: str="ndvi"
) -> None:
    """
    This function extracts long-term NDVI and LST time-series from MODIS
    at a given resolution.
    """
    
    # MODIS band data need to be scaled appropriately
    if index_type == "ndvi":
        dataset = "MODIS/061/MOD13Q1" # MODIS NDVI dataset

        def preprocess(img: ee.Image) -> ee.Image:
            img_scaled = img.select("NDVI").multiply(0.0001).rename("ndvi")
            return img_scaled.copyProperties(img, ["system:time_start"])
    elif index_type == "lst":
        dataset = "MODIS/061/MOD11A2" # MODIS LST dataset

        def preprocess(img: ee.Image) -> ee.Image:
            img_scaled = img.select("LST_Day_1km").multiply(0.02).subtract(273.15).rename("lst")
            return img_scaled.copyProperties(img, ["system:time_start"])
    elif index_type == "precipitation":
        dataset = "UCSB-CHG/CHIRPS/PENTAD"

        def preprocess(img: ee.Image) -> ee.Image:
            img = img.select("precipitation").rename("precipitation")
            return img.copyProperties(img, ["system:time_start"])
    elif index_type == "pdsi":
        dataset = "IDAHO_EPSCOR/TERRACLIMATE"

        def preprocess(img: ee.Image) -> ee.Image:
            img_scaled = img.select("pdsi").multiply(0.01)
            return img_scaled.copyProperties(img, ["system:time_start"])

    else:
        logging.error(f"Unsupported index type: {index_type}") 

    img_collection = (
        ee.ImageCollection(dataset)
        .filterDate(START_DATE, END_DATE)
        .filterBounds(FeatureCollection)
        .map(preprocess)
    )

    def map_index(img: ee.Image):
        """ 
        This function applies a reducer to each image in a collection and
        performs a spatial mean of the NDVI.
        """
        stats = img.reduceRegions(
            collection=FeatureCollection,
            reducer=ee.Reducer.mean().setOutputs([index_type]),
            scale=1000
        ).filter(ee.Filter.neq(index_type, None)) # Filter out null NDVI values

        def set_date(feature: ee.Feature):
            return feature.set("date", img.date().format("YYYY-MM-dd"))
        
        return stats.map(set_date)
    
    return img_collection.map(map_index).flatten()

if __name__ == "__main__":
    gdf = gpd.read_file("../inference/Trans_Nzoia_1_results_aggregated.gpkg")
    grid = generate_pixel_grid(gdf, "Trans_Nzoia_1")
    
    grid["RoI"] = grid["geometry"].apply(return_ee_geometry)
    geometries = gdf_to_FeatureCollection(grid)

    index_type = "pdsi"
    img_collection = generate_MODIS_timeseries(geometries, index_type=index_type)
    index_table = format_table(img_collection, "tile_name", "date", index_type)

    table_name = f"MODIS_{index_type}_series_Trans_Nzoia_1"

    # Export as a table
    task = ee.batch.Export.table.toDrive(
        collection=index_table,
        description=table_name,
        folder="MODIS_time_series",
        fileFormat="CSV"
    )
    task.start()