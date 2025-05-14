""" 
Scripts for generating clean ndvi time-series data
"""
#import numpy as np
import pandas as pd 
from preprocessing import clean_ndvi_series

def clean_train_samples(
        region: str,
        tile: int | list[int],
        input_dir: str="ndvi_series_raw",
        output_dir: str="ndvi_series_labeled"
) -> None:
    """ 
    This function cleans a few tiles of NDVI data for the 
    purposes of training classifiers. The data are read from
    the `ndvi_series_raw` directory.
    """
    if isinstance(tile, int):
        tile = [tile]

    dfs = []
    for t in tile:
        fname = f"ndvi_series_{region}_tile_{t}.csv"
        print(f"Tile_{t}")
        df_tile = pd.read_csv(f"{input_dir}/{fname}")
        df_tile = clean_ndvi_series(df_tile)
        dfs.append(df_tile)

    df_train = (
        pd.concat(dfs)
        .reset_index(drop=True)
    )

    df_train.to_csv(f"{output_dir}/ndvi_series_{region}_clean.csv")