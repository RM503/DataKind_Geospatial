''' 
Main file for requesting and saving Sentinel Hub images.
'''
import os
import logging
from tqdm import tqdm
import numpy as np
import pandas as pd 
import geopandas as gpd 
from shapely import wkt
from utils import (
    generate_covering_grid,
    SH_request_builder,
    edge_enhance,
    array_to_geotiff
)

logging.basicConfig(
    level=logging.INFO, 
    filename='logs/main_log.txt', 
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def append_running_count(
    gdf: gpd.GeoDataFrame,
    column_name: str,
    new_column_name: str
) -> gpd.GeoDataFrame:
    '''
    This function appends a running count to `County` names. This is done such that we are able to
    distinguish between distributors in the same county.

    Args: (i) gdf - geopandas dataframe containing distributors
          (ii) column_name - `County` column name
          (iii) new_column_name - name of the new column which contains the count information

    Returns: new geopandas dataframe with count information
    '''

    gdf['running_count'] = gdf.groupby(column_name).cumcount() + 1
    gdf[new_column_name] = gdf.apply(
        lambda row: f"{row[column_name]} {row['running_count']}", axis=1
    )

    gdf.drop(columns=['running_count'], inplace=True)
    return gdf

def main(gdf: gpd.GeoDataFrame, img_type: str) -> None:
    ''' 
    This function generates image tiles for each distributor location by using
    Sentinel Hub's request builder API.
    '''
    # Generate covering grids for each location as MultiPolygons
    gdf = gdf.drop(columns=['#', 'Unnamed: 0'])
    gdf_w_grids = generate_covering_grid(gdf)

    if 'tile_grids' not in gdf_w_grids.columns.tolist():
        logging.ERROR("'tilde_grids' column not present; tiles cannot be requested from SentinelHub")
    
    '''
    The `County` information will be used to identify GeoTIFF tiles. However,
    county information may be repeated. This is fixed by adding a running count column
    to the dataframe.
    '''
    
    gdf_w_grids = append_running_count(gdf_w_grids, 'County', 'County_enumerated')
    for idx, row in tqdm(gdf_w_grids.iterrows(), desc='Iterating over locations'):
        print(f"Current location: {row['County_enumerated']}")

        FOLDER_NAME = '_'.join(row['County_enumerated'].split(' '))
        EXPORT_DIR = 'images/' + FOLDER_NAME + '/' 

        tiles = row['tile_grids']
        START_DATE = '2024/01/01'
        END_DATE = '2025/01/01'

        if img_type == 'PNG':
             # Tile-image generator object
            tile_imgs = SH_request_builder(
                    tiles=tiles,
                    start_date=START_DATE,
                    end_date=END_DATE,
                    export_dir=EXPORT_DIR,
                    evalscript='highlight_optimized',
                    img_type=img_type
                )
            for count, (img, lon_array, lat_array) in enumerate(tile_imgs):
                # Apply edge enhancement and export as GeoTIFF
                img = edge_enhance(img)
                IMG_PATH = EXPORT_DIR + 'tile_' + str(count) + '.tiff'

                if not os.path.exists(EXPORT_DIR):
                    logging.INFO(f'Creating directory: {EXPORT_DIR}')
                    os.makedirs(EXPORT_DIR)

                array_to_geotiff(img, IMG_PATH, lat_array, lon_array)
        elif img_type == 'TIFF':
             # Tile-image generator object
            tile_imgs = SH_request_builder(
                    tiles=tiles,
                    start_date=START_DATE,
                    end_date=END_DATE,
                    export_dir=EXPORT_DIR,
                    evalscript_type='highlight_optimized',
                    img_type=img_type
                )
            for count, request in enumerate(tile_imgs):
                IMG_NAME = 'tile_' + str(count) + '.tiff'
                
                if not os.path.exists(EXPORT_DIR):
                    logging.INFO(f'Creating directory: {EXPORT_DIR}')
                    os.makedirs(EXPORT_DIR)
                request.download_list[0].filename = IMG_NAME
                request.save_data()

if __name__ == '__main__':

    data_df = pd.read_csv('../../geocoding/distributor_locations_priority_geocoded.csv')
    data_df.geometry = data_df.geometry.apply(wkt.loads)
    data_gdf = gpd.GeoDataFrame(data_df)

    main(data_gdf, 'TIFF')