from pathlib import Path
from src.segmentation.s3_io import (
    TileJob,
    build_tile_jobs,
    discover_regions,
    download_file,
    list_all_keys,
    list_region_tile_keys,
    make_s3_client,
    object_exists
)

BUCKET_NAME = "regenorganics-prioritydistributorbuffer-tiffs"

if __name__ == "__main__":
    
    client = make_s3_client(region_name="us-east-1")
    
    print(object_exists(client, BUCKET_NAME, "Trans_Nzoia_1/tile_0.tiff"))