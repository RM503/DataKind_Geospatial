"""
Scripts for performing postprocessing on inference results
"""
import glob
import pandas as pd 
import geopandas as gpd 
from sktime.datatypes._panel._convert import from_long_to_nested

def ts_long_to_nested(file_paths: list[str]) -> pd.DataFrame:
    """ 
    This function converts the NDVI time-series data from long to a nested
    format, which can be stored alongside polygon data.

    Args: (i) file_paths - file paths to clean NDVI time-series

    Returns: df_nested - nested dataframe
    """
    df_list = [pd.read_csv(path, parse_dates=["date"]) for path in file_paths]
    df = pd.concat(df_list).reset_index(drop=True)

    if df["date"].dtype != "datetime64[ns]":
        df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")

    # sktime `from_long_to_nested` expects multivariate data; requires a `dim_id`
    if "dim_id" not in df.columns:
  
        df["dim_id"] = "ndvi"

    df_nested = from_long_to_nested(
        df,
        instance_column_name="uuid",
        time_column_name="date",
        value_column_name="ndvi"
    ).reset_index()

    # The value column will have a default name `var_0`
    df_nested = df_nested.rename(columns={"var_0": "ndvi_time_series"})

    return df_nested
        
def merge_df_gdf(
        df: pd.DataFrame,
        gdf: gpd.GeoDataFrame,
        df_ts_nested: pd.DataFrame
) -> gpd.GeoDataFrame:
    """ 
    This function merges the NDVI time-series data from inferentia with
    corresponding .gpkg file for a given region by `uuid`. The function then
    filters the resulting dataframe by keeping only `Farm` and `Field` polygons
    while also filtering out the very small ones. Additionally, the NDVI time-series
    data are also attached as an additional column.

    Args: (i) df - inference data for a given region
          (ii) gdf - polygon file for a given region
          (iii) df_ts_nested - the NDVI data for given region in the sktime nested format
    
    Returns: gdf_merged - a merged geodataframe with predictions and nested time-series
                          for polygons
    """

    df_merged = (
        df.merge(gdf, on="uuid", how="inner")
          .merge(df_ts_nested, on="uuid", how="inner")
    )

    # Filter out the very small polygons that were used in training and keep only `Farm` and `Field`
    filter_condition = (df_merged["area (acres)"] >= 0.25) & (df_merged["prediction_decoded"].isin(["Farm", "Field"]))
    df_filtered = df_merged[filter_condition].reset_index(drop=True)

    return gpd.GeoDataFrame(df_filtered, geometry="geometry")

def main(region: str) -> None:
    """
    This function stitches together the two preceeding steps.
    """
    # Retrieve list of all cleaned NDVI time-series for given region
    ndvi_file_paths = glob.glob(f"../ndvi_series_clean/ndvi_series_{region}_aggregated.csv")

    df_ts_nested = ts_long_to_nested(ndvi_file_paths)

    # Retrive .gpkg polygon file for given region
    gdf = gpd.read_file(f"../../../samgeo_aws_ec2/vectors/{region}.gpkg")

    # Rerieve prediction dataframe for given region
    df = pd.read_csv(f"../inference/ndvi_series_{region}_predictions.csv")

    gdf_merged = merge_df_gdf(df, gdf, df_ts_nested)
    gdf_merged.to_file(f"../inference/{region}_results_aggregated.gpkg", driver="GPKG")