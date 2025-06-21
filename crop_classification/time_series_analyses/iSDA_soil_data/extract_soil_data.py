import glob
import boto3
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import rasterio as rio
import rasterio.features
from rasterstats import zonal_stats
import logging 

logging.basicConfig(level=logging.INFO)

def data_within_polygons(gdf: gpd.GeoDataFrame, raster_file_paths: list[str], region: str="Trans_Nzoia_1") -> gpd.GeoDataFrame:
    # Making a copy so that the original is not changed
    if not isinstance(gdf, gpd.GeoDataFrame):
        logging.error("gdf is not a GeoDataFrame")
        
    gdf_copied = (
        gdf.copy(deep=True)
        .reset_index(drop=True)
    )

    for file_path in sorted(raster_file_paths):
        qty_name = file_path.split("/")[-1].split(f"_{region}.tif")[0] # get name of soil quantity
        logging.info(f"Working on {qty_name}")

        """ 
        We now calculate the zonal statistics using rasterstats. Looking through
        the rows of the gdf is inefficient.
        """
        with rio.open(file_path, mode="r") as src:
            data = src.read(1) 
            transform = src.transform
            
            stats = zonal_stats(
                gdf_copied,
                data,
                affine=transform,
                stats=["mean"]
            )
        means = [s["mean"] if s["mean"] is not None else np.nan for s in stats]
        gdf_copied[qty_name] = means

    return gdf_copied