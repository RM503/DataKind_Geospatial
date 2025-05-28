"""
Scripts for generating clean NDVI time-series data from raw ones
by using preprocessing codes
"""

import pandas as pd
import data_processing.preprocessing as preprocessing
import glob 
from concurrent.futures import ProcessPoolExecutor 
import logging

logging.basicConfig(level=logging.INFO)

def get_file_paths(regions: str | list[str], input_dir: str="ndvi_series_raw") -> list[str]:

    # Initialize an empty list to store file directories
    all_files = []
    for region in regions:
        INPUT_FILE_PATH = f"{input_dir}/ndvi_series_{region}_*.csv"
        region_files = glob.glob(INPUT_FILE_PATH)

        all_files.extend(region_files)

    return all_files

def clean_data(input_file_path: str, output_dir: str="ndvi_series_clean") -> None:
    logging.info(f"Processing file: {input_file_path}")

    df_tile = pd.read_csv(input_file_path)
    df_tile_cleaned = preprocessing.clean_vi_series(df_tile, vi="ndvi", date_resample=False)

    file_name = input_file_path.split("/")[-1].split(".")[0]
    OUTPUT_FILE_PATH = f"{output_dir}/{file_name}_clean.csv"

    df_tile_cleaned.to_csv(OUTPUT_FILE_PATH, index=False)

if __name__ == "__main__":
    import argparse 

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--regions",
        nargs="+",
        required=True,
        help="Enter the region(s) for cleaning. If multiple regions, enter a list."
    )

    args = parser.parse_args()
    regions = args.regions

    file_paths = get_file_paths(regions)

    # Spawn multiple processes using ProcessPoolExecutor context manager
    with ProcessPoolExecutor(max_workers=4) as executor:
        for path in file_paths:
            logging.info(f"Spawning process for {path}")
            executor.submit(clean_data, path)