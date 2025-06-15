""" 
Converts .gpkg to .csv and .kml files
"""
import glob
import geopandas as gpd
import fiona
fiona.supported_drivers['KML'] = 'rw'

def convert_to_csv(gpkg_folder_path: str) -> None:
    if gpkg_folder_path.endswith('/'):
        gpkg_folder_path = gpkg_folder_path[:-1]
        
    gpkg_list = glob.glob(f"{gpkg_folder_path}/*.gpkg")
    if len(gpkg_list) == 0:
        raise ValueError(f"No .gpkg files found in {gpkg_folder_path}")

    for gpkg in gpkg_list:
        gdf = gpd.read_file(gpkg)

        csv_file_name = f"{gpkg.split('/')[-1].split('.')[0]}.csv"
        gdf.to_csv(f"{gpkg_folder_path}/{csv_file_name}", index=False)

def convert_to_kml(gpkg_folder_path: str) -> None:
    if gpkg_folder_path.endswith('/'):
        gpkg_folder_path = gpkg_folder_path[:-1]
        
    gpkg_list = glob.glob(f"{gpkg_folder_path}/*.gpkg")
    if len(gpkg_list) == 0:
        raise ValueError(f"No .gpkg files found in {gpkg_folder_path}")

    for gpkg in gpkg_list:
        gdf = gpd.read_file(gpkg)

        kml_file_name = f"{gpkg.split('/')[-1].split('.')[0]}.kml"
        gdf.to_file(f"{gpkg_folder_path}/{kml_file_name}", driver='KML')
    

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpkg_folder_path', required=True)

    arg = parser.parse_args()
    gpkg_folder_path = arg.gpkg_folder_path

    convert_to_kml(gpkg_folder_path)