import glob 
import pandas as pd
import logging 

logging.basicConfig(level=logging.INFO)

def aggregate_ts(
        region: str,
        vi: str="ndvi",
        input_file_dir: str="../ndvi_series_clean/",
        output_file_dir: str=None
) -> None:
    """ 
    This function aggregates the time-series files from all the tiles of a given region.

    Args: (i) region - name of the region; must be among one of the regions below
          (ii) input_file_dir - directory containing the clean time-series files
          (iii) output_file_dir - directory where the aggregrated files are exported; uses input_file_dir if None

    Returns: None
    """
    if region not in ["Kajiado_1", "Kajiado_2", "Laikipia_1", "Machakos_1", "Mashuru_1", "Trans_Nzoia_1"]:
        logging.error(f"Invalid region: {region}")

    base_file_name = f"{vi}_series"
    file_paths = glob.glob(f"{input_file_dir}{base_file_name}_{region}_*.csv")
    if len(file_paths) == 0:
        logging.error(f"No files found for region {region} matching required pattern.")

    df_list = []
    for file in file_paths:
        # Try to read with date parsing
        df = pd.read_csv(file, parse_dates=["date"])  
        df = df.loc[:, ~df.columns.str.startswith("Unnamed:")] # Get rid of all `Unnamed:` index columns
        df["date"] = pd.to_datetime(df["date"], format="mixed", errors="coerce")
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")  # Save as string in 'YYYY-MM-DD' format
        df_list.append(df)

    # Ensure columns are consistent
    base_columns = df_list[0].columns
    for i, df in enumerate(df_list):
        if not df.columns.equals(base_columns):
            print(f"Checking file: {file_paths[i]}")
            print(f"⚠️ Column mismatch in file: {file_paths[i]}")
            print(f"Columns found: {df.columns.tolist()}")
            print(f"Expected: {base_columns.tolist()}")
            return

    df = pd.concat(df_list).reset_index(drop=True)

    if output_file_dir is not None:
        df.to_csv(f"{output_file_dir}{base_file_name}_{region}_aggregated.csv")
    else:
        df.to_csv(f"{input_file_dir}{base_file_name}_{region}_aggregated.csv") 
