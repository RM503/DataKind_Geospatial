import numpy as np
import pandas as pd 
import geopandas as gpd 
from shapely import wkt
from utils import (
    get_covering_grid,
    SH_request_builder,
    edge_enhance
)
from samgeo import array_to_image

data_df = pd.read_csv('../../geocoding/distributor_locations_priority_geocoded.csv')
data_df.geometry = data_df.geometry.apply(wkt.loads)
data_gdf = gpd.GeoDataFrame(data_df)

data_w_grids = get_covering_grid(data_gdf)

