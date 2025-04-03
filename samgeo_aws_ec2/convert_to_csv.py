""" 
Converts .gpkg to .csv
"""
import glob
import pandas as pd 
import geopandas as gpd
import argparse

def convert_to_csv(gpkg_folder_path: str) -> None:
    if gpkg_folder_path.endswith('/'):
        gpkg_folder_path = gpkg_folder_path[:-1]
        
    gpkg_list = glob.glob(f"{gpkg_folder_path}/*.gpkg")

    for gpkg in gpkg_list:
        gdf = gpd.read_file(gpkg)

        csv_file_name = gpkg.split('/')[-1].split('.')[0] + '.csv'
        gdf.to_csv(f"{gpkg_folder_path}/{csv_file_name}", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpkg_folder_path', required=True)

    arg = parser.parse_args()
    gpkg_folder_path = arg.gpkg_folder_path

    convert_to_csv(gpkg_folder_path)