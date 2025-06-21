"""
Scripts for generating clean VI time-series data from raw ones
by using preprocessing codes
"""
import os
import pandas as pd
import data_processing.preprocessing as preprocessing
import glob 
from concurrent.futures import ProcessPoolExecutor 
import logging

logging.basicConfig(level=logging.INFO)

def get_file_paths(regions: list[str], vi: str="ndvi", input_dir: str="ndvi_series_raw") -> list[str]:
    """ 
    This function obtains a list of file paths for a given VI type and region.

    Args: (i) regions - a list (singlet or other) of regions
          (ii) vi - the VI type; defaults to `NDVI`
          (iii) input_dir - the input directory; defaults to `ndvi_series_raw`
    
    Returns: a list of file paths
    """

    # Initialize an empty list to store file directories
    all_files = []

    base_file_name = f"{vi}_series"
    for region in regions:
        INPUT_FILE_PATH = f"{input_dir}/{base_file_name}_{region}_*.csv"
        region_files = glob.glob(INPUT_FILE_PATH)

        if len(region_files) == 0:
            logging.warning(f"No files found for region: {region}")
            continue
        
        # Extend all glob lists into a single list
        all_files.extend(region_files)

    return all_files

def clean_data(input_file_path: str, vi: str="ndvi", output_dir: str="ndvi_series_clean") -> None:
    """ 
    This function cleans raw data provided by applying the preprocessing steps in `data_processing/preprocessing.py`
    and saves them to the specified output directory.

    Args: (i) input_file_path - the input file path
          (ii) vi - the VI type; defaults to `NDVI`
          (iii) output_dir - the output directory       

    Returns: None
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    logging.info(f"Processing file: {input_file_path}")

    df_tile = pd.read_csv(input_file_path, parse_dates=["date"])
    df_tile_cleaned = preprocessing.clean_vi_series(df_tile, vi=vi, resample_date=True)

    file_name = input_file_path.split("/")[-1].split(".")[0]
    OUTPUT_FILE_PATH = f"{output_dir}/{file_name}_clean.csv"

    df_tile_cleaned.to_csv(OUTPUT_FILE_PATH, index=False)

if __name__ == "__main__":
    import argparse 

    # Use argparse to pass in the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--regions",
        nargs="+",
        required=True,
        help="Enter the region(s) for cleaning. If multiple regions, enter a list."
    )
    parser.add_argument(
        "--vi",
        type=str,
        default="ndvi",
        help="Specify VI type"
    )
    args = parser.parse_args()
    regions = args.regions
    vi = args.vi

    file_paths = get_file_paths(regions, vi=vi, input_dir="ndmi_series_raw")

    if vi == "ndvi":
        output_dir = "ndvi_series_clean"
    elif vi == "ndmi":
        output_dir = "ndmi_series_clean"

    # Spawn multiple processes using ProcessPoolExecutor context manager
    with ProcessPoolExecutor(max_workers=4) as executor:
        for path in file_paths:
            logging.info(f"Spawning process for {path}")
            executor.submit(clean_data, path, vi, output_dir)