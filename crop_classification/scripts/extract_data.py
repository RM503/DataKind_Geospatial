import pandas as pd
import geopandas as gpd
import dask.dataframe as dd
import geopandas as gpd
from scipy.signal import savgol_filter
import logging 

logging.basicConfig(level=logging.INFO)

def extract_data(
        ddf_ndvi: dd.DataFrame, 
        ddf_ndmi: dd.DataFrame, 
        gdf: gpd.GeoDataFrame,
        smooth: bool=True
    ) -> pd.DataFrame:
    """
    This function extracts all time-series information from cleaned NDVI and NDMI data.
    Since the data can potentially create memory related problems, we use Dask

    Args: (i) ddf_ndvi - the NDVI dask dataframe
          (ii) ddf_ndmi - the NDMI dask dataframe
          (iii) gdf - geodataframe with polygon information (aggregated)

    Returns: a dataframe containing both NDVI and NDMI columns
    """
    if not isinstance(ddf_ndvi, dd.DataFrame) or not isinstance(ddf_ndmi, dd.DataFrame):
        logging.warning(f"Either one or both dataframes are not {dd.DataFrame} types. This can create memory related issues")

    uuid_list = gdf["uuid"].tolist()
    polygon_type_list = gdf["prediction_decoded"].tolist()

    # Merge NDVI and NDMI data and create a polygon type mapping
    df_ndvi = ddf_ndvi[ddf_ndvi.uuid.isin(uuid_list)].compute()
    df_ndmi = ddf_ndmi[ddf_ndmi.uuid.isin(uuid_list)].compute()

    df_ndvi = df_ndvi.loc[:, ~df_ndvi.columns.str.startswith("Unnamed:")]
    df_ndmi = df_ndmi.loc[:, ~df_ndmi.columns.str.startswith("Unnamed:")]

    df_merged = df_ndvi.merge(df_ndmi, on="uuid", how="inner")

    polygon_mapping = dict(zip(uuid_list, polygon_type_list))
    df_merged["polygon_type"] = df_merged.groupby("uuid")["uuid"].transform(lambda x: polygon_mapping[x.iloc[0]])

    # Further smoothing may be required to remove unwanted peaks
    if smooth:
        df_merged["ndvi"] = df_merged.groupby("uuid")["ndvi"].transform(
            lambda x: savgol_filter(x, 7, 3)
        )

        df_merged["ndmi"] = df_merged.groupby("uuid")["ndmi"].transform(
            lambda x: savgol_filter(x, 7, 3)
        )

        return df_merged
    return df_merged
