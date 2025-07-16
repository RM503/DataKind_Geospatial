import geopandas as gpd
import dask.dataframe as dd 
from scripts.extract_data import extract_data
from scripts.farm_characteristics import ExtractNDVIData, ExtractNDMIData

import logging 

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    regions = ["Kajiado_1", "Kajiado_2", "Laikipia_1", "Trans_Nzoia_1"]

    for region in regions: 
        logging.info(f"Working on region: {region}")

        FARM_POLYGON_FILEPATH = f"time_series_analyses/inference/{region}_results_aggregated.gpkg"
        NDVI_FILEPATH = f"time_series_analyses/ndvi_series_clean/ndvi_series_{region}_aggregated.csv"
        NDMI_FILEPATH = f"time_series_analyses/ndmi_series_clean/ndmi_series_{region}_aggregated.csv"
        
        # Read time-series into Dask dataframes for efficient handling
        dd_ndvi = dd.read_csv(NDVI_FILEPATH, parse_dates=["date"])
        dd_ndmi = dd.read_csv(NDMI_FILEPATH, parse_dates=["date"])
        gdf = gpd.read_file(FARM_POLYGON_FILEPATH)

        df_merged = extract_data(dd_ndvi, dd_ndmi, gdf)

        ndvi_data_extractor = ExtractNDVIData(df_merged, region)
        
        df_peaks = ndvi_data_extractor.ndvi_peaks()

        _, _ = ndvi_data_extractor.ndvi_peak_annual_dists(df_peaks)

        ndmi_data_extractor = ExtractNDMIData(df_merged, region)
        _ = ndmi_data_extractor.high_ndmi_days()
        _ = ndmi_data_extractor.moisture_content()