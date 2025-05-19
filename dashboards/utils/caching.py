import os
import geopandas as gpd

def create_caches(data_group: str, region: str) -> None:
    CACHE_DIR = f"../assets/{data_group}/cached"

    if not os.path.isdir(CACHE_DIR):
        os.makedirs(CACHE_DIR)

    if data_group == "raw":
        
        gdf = gpd.read_file(f"../assets/{data_group}/{region}.gpkg")
        gdf.to_feather(f"{CACHE_DIR}/{region}.feather")

    if data_group == "processed":
        gdf = gpd.read_file(f"../assets/{data_group}/{region}_results_aggregated.gpkg")
        gdf.to_feather(f"{CACHE_DIR}/{region}_results_aggregated.feather")


if __name__ == "__main__":
    create_caches("raw", "Laikipia_1")
    create_caches("raw", "Trans_Nzoia_1")
    create_caches("processed", "Laikipia_1")
    create_caches("processed", "Trans_Nzoia_1")