import os
import glob
from urllib.parse import urlparse
import geopandas as gpd
from pystac import Catalog
from pystac.stac_io import StacIO
import boto3 
from botocore.exceptions import ClientError  
import pyproj
from scripts.isda_data_extract import ISDADataExtract, data_within_polygons
import logging 

logging.basicConfig(level=logging.INFO)

os.environ["PROJ_DATA"] = pyproj.datadir.get_data_dir()

def my_read_method(uri):
    # This function is used to read iSDA data from `https://registry.opendata.aws/isdasoil`.

    parsed = urlparse(uri)
    if parsed.scheme == "s3":
        bucket = parsed.netloc
        key = parsed.path[1:]

        try:
            s3 = boto3.resource("s3") 
            obj = s3.Object(bucket, key) 

            return obj.get()["Body"].read().decode("utf-8")
        except ClientError as e:
            logging.error(f"Error connecting to S3 bucker: {e}")

    else:
        return StacIO.default_read_text_method(uri)


if __name__ == "__main__":
    regions= ["Kajiado_1", "Kajiado_2", "Laikipia_1", "Trans_Nzoia_1"]
    qty_types = [
        "bulk_density",
        "calcium_extractable",
        "carbon_organic",
        "carbon_total",
        "clay_content",
        "iron_extractable",
        "magnesium_extractable",
        "nitrogen_total",
        "ph",
        "phosphorous_extractable",
        "potassium_extractable",
        "sand_content",
        "silt_content",
        "stone_content",
        "sulphur_extractable",
        "texture_class",
        "zinc_extractable"
    ]
    
    StacIO.read_text_method = my_read_method 
    catalog = Catalog.from_file("https://isdasoil.s3.amazonaws.com/catalog.json")

    assets = {}

    for root, catalogs, items in catalog.walk():
        for item in items:
            print(f"Type: {item.get_parent().title}")
            # save all items to a dictionary as we go along
            assets[item.id] = item
            for asset in item.assets.values():
                if asset.roles == ['data']:
                    print(f"Title: {asset.title}")
                    print(f"Description: {asset.description}")
                    print(f"URL: {asset.href}")
                print("------------")

    for region in regions:
        logging.info(f"Working on {region}")

        gdf = gpd.read_file(f"time_series_analyses/inference/{region}_results_aggregated.gpkg")
        gdf_farm = gdf[gdf["prediction_decoded"]=="Farm"]

        xmin, ymin, xmax, ymax = gdf.total_bounds
        # Upper left (ymax, xmin) and lower right corners (ymin, xmax)
        start_lat_lon = (ymax, xmin)
        end_lat_lon = (ymin, xmax)

        for qty_type in qty_types:
            logging.info(f"Working on {qty_type}")
            isda = ISDADataExtract(region, start_lat_lon, end_lat_lon, assets, qty_type)
            data, _, X, Y = isda.get_data_subset() 

        # Generate data table from rasters
        raster_paths = sorted(glob.glob(f"isda_data/{region}.tif"))
        data_within_polygons(
            gdf_farm,
            raster_paths,
            region,
            return_data=False
        )