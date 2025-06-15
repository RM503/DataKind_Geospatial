import os
import geopandas as gpd
from shapely.geometry import base 
import fiona 
import logging
fiona.supported_drivers["KML"] = "rw" 
logging.basicConfig(level=logging.INFO)

def convert_to_csv_and_kml(
        file_path: str,
        export_dir: str,
        simplify: bool=True,
        tolerance: float=1e-4
) -> None:
    """ 
    Exports aggregated .gpkg files to .csv and .kml formats, where data of each
    region is split by tile number.
    """
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)

    gdf = gpd.read_file(file_path)

    if simplify:
        gdf["geometry"] = gdf["geometry"].simplify(tolerance=tolerance, preserve_topology=True)

    # Drop unneeded columns if present
    columns_to_drop = ["prediction", "area (sq. meters)", "ndvi_time_series"]
    gdf.drop(columns=[col for col in columns_to_drop if col in gdf.columns], inplace=True)

    # Rename `prediction_decoded` column
    if "prediction_decoded" in gdf.columns:
        gdf.rename(columns={"prediction_decoded": "polygon_type"}, inplace=True)

    # Group data by tile_name
    for idx, group in gdf.groupby("tile_name"):
        logging.info(f"Exporting tile {idx}")

        # Remove all columns not needed in the KML (optional)
        kml_gdf = group[["geometry", "uuid", "polygon_type"] + [col for col in gdf.columns if col not in ["geometry", "uuid", "polygon_type"]]]

        # Ensure correct CRS for KML
        if kml_gdf.crs != "EPSG:4326":
            kml_gdf = kml_gdf.to_crs("EPSG:4326")

        export_file_name = f"{file_path.split('/')[-1].split('.')[0]}_tile_{idx}"
        export_file_path = f"{export_dir}/{export_file_name}"

        kml_gdf.to_file(f"{export_file_path}.kml", driver="KML")

        csv_gdf = group.copy()
        csv_gdf["geometry"] = csv_gdf["geometry"].apply(lambda x: x.wkt if isinstance(x, base.BaseGeometry) else None)

        csv_gdf.to_csv(f"{export_file_path}.csv", index=False)

if __name__ == "__main__":
    file_path = "../inference/Kajiado_1_results_aggregated.gpkg"
    export_dir = "../inference/for_regen"
    convert_to_csv_and_kml(file_path, export_dir)