import os
import geopandas as gpd

def create_caches(data_group: str, region: str) -> None:
    """ 
    Creates lighter .feather files with simplified geometries for
    faster loading
    """
    CACHE_DIR = f"../assets/{data_group}/cached"

    if not os.path.isdir(CACHE_DIR):
        os.makedirs(CACHE_DIR)

    if data_group == "raw":
        
        gdf = gpd.read_file(f"../assets/{data_group}/{region}.gpkg")
        gdf["geometry"] = gdf["geometry"].simplify(tolerance=1e-4, preserve_topology=True)
        gdf.to_feather(f"{CACHE_DIR}/{region}.feather")

    if data_group == "processed":
        gdf = gpd.read_file(f"../assets/{data_group}/{region}_results_aggregated.gpkg")
        gdf["geometry"] = gdf["geometry"].simplify(tolerance=1e-4, preserve_topology=True)
        gdf.to_feather(f"{CACHE_DIR}/{region}_results_aggregated.feather")

if __name__ == "__main__":

    regions = ["Kajiado_2", "Machakos_1", "Mashuru_1"]

    for region in regions:
        create_caches("raw", region)
        create_caches("processed", region)