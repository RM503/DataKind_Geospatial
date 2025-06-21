import os
import numpy as np
import pandas as pd
import pandera as pa
from pandera.typing.pandas import Series
from pandera.errors import SchemaError
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import seaborn as sns
import logging 

sns.set_palette("bright")
logging.basicConfig(level=logging.INFO)

class Schema(pd.SchemaModel):
    # Schema enforcement
    date: Series[pa.DateTime] = pa.Field(nullable=False)
    uuid: Series[str] = pd.Field(nullable=False)
    ndvi: Series[float] = pd.Field(ge=-1.0, le=1.0, nullable=False)
    ndmi: Series[float] = pd.Field(ge=-1.0, le=1.0, nullable=False)
    polygon_type: Series[str] = pd.Field(isin=["Farm", "Field"], nullable=False)

class ExtractNDVIData:
    def __init__(self, df: pd.DataFrame, region: str, validate_input: bool=True):
        # Check if input data has valid schema
        if validate_input:
            try:
                df = Schema(df)
            except SchemaError as e:
                logging.error("Data validation failed!")
                raise e
            
        self.df = df
        self.df_prepared = self._prepare_data() 
        self.region = region

    def _prepare_data(self) -> pd.DataFrame:
        df_copy = self.df.copy(deep=True)

        try:
            df_copy = df_copy[df_copy["polygon_type"]=="Farm"]
        except KeyError as e:
            logging.error("polygon_type column is not present in the dataframe.")
            raise e
        
        if not pd.api.types.is_datetime64_any_dtype(df_copy["date"]):
            df_copy["date"] = pd.to_datetime(df_copy["date"])

        return df_copy

    def ndvi_peaks(self, export: bool=True) -> pd.DataFrame:
        """ 
        This function groups NDVI time-series data by uuid and applies peak-finding algorithm
        in order to identify NDVI peaks occurring in each identified farm.
        """
        df = self.df_prepared

        try:
            df = df[df["polygon_type"]=="Farm"]
        except KeyError as e:
            logging.error("polygon_type column is not present in the dataframe.")
            raise e

        peaks_date_dict = {} # dictionary for storing dates where NDVI peaks occur for uuid
        peaks_val_dict = {}

        # Group by uuid
        for uuid, group in df.groupby("uuid"):
            group = group.reset_index(drop=True)
            peaks, _ = find_peaks(
                group["ndvi"].values, 
                height=(0.4, 1.0), 
                prominence=0.10, 
                distance=10
            )
            group["peak"] = np.isin(group.index, peaks).astype(int)
            
            # Extract dates where NDVI peaks occur and corresponding NDVI values
            ndvi_peak_dates = group[group["peak"] == 1]["date"].tolist()
            ndvi_peak_values = group[group["peak"] == 1]["ndvi"].tolist()

            peaks_date_dict[uuid] = ndvi_peak_dates    
            peaks_val_dict[uuid] = ndvi_peak_values

        """ 
        peaks_dict contains arrays of unequal lengths and, as such, cannot be used
        to construct a dataframe. Hence, we need to pad all arrays to the same length.
        """
        max_len = max(len(v) for v in peaks_date_dict.values()) # Find length of longest entry

        # Padding all arrays to the same length
        peaks_date_dict_padded = {
            k: v + [None]*(max_len - len(v)) for k, v in peaks_date_dict.items()
        }
        df_peaks_date = pd.DataFrame(peaks_date_dict_padded)

        peaks_val_dict_padded = {
            k: v + [None]*(max_len - len(v)) for k, v in peaks_val_dict.items()
        }
        df_peaks_val = pd.DataFrame(peaks_val_dict_padded)

        # Convert the wide form df to long form
        df_peaks_date["index"] = range(max_len)
        df_peaks_val["index"] = range(max_len)

        df_peaks_date_melted = pd.melt(
            df_peaks_date, 
            id_vars="index", 
            var_name="uuid", 
            value_name="ndvi_peak_date"
        ).dropna()

        df_peaks_val_melted = pd.melt(
            df_peaks_val, 
            id_vars="index", 
            var_name="uuid", 
            value_name="ndvi_peak_value"
        ).dropna()

        df_merged = df_peaks_date_melted.merge(df_peaks_val_melted, on=["uuid", "index"], how="inner")
        df_merged = df_merged.drop(columns="index")

        # Add a cumulative count to assign an order label to the peaks (eg. 1st, 2nd, 3rd,... peak of the year)
        df_merged["year"] = df_merged["ndvi_peak_date"].dt.year 
        df_merged["peak_position"] = (df_merged.groupby(["uuid", "year"]).cumcount() + 1).astype("int")
        df_merged.drop(columns=["year"], inplace=True)

        if export:
            if not os.path.exists(f"vi_characteristics/{self.region}"):
                logging.info("Creating export directory.")
                os.makedirs(f"vi_characteristics/{self.region}", exist_ok=True)
            
            df_merged.to_csv(f"vi_characteristics/{self.region}/ndvi_peaks_{self.region}.csv", index=False)

            # Export plots 
            if not os.path.exists(f"plots/{self.region}"):
                os.makedirs(f"plots/{self.region}", exist_ok=True)

            sns.boxplot(df_merged, x="peak_position", y="ndvi_peak_value", hue="peak_position", edgecolor="k")
            plt.xlabel("Position", fontsize=12)
            plt.ylabel("NDVI peak value", fontsize=12)
            plt.title("NDVI peaks in relation to occurrence during year", fontsize=14)
            plt.savefig(f"plots/{self.region}/ndvi_peaks_boxplot_{self.region}.png", dpi=300, bbox_inches="tight")

        return df_merged

    def ndvi_peak_annual_dists(self, df_peaks: pd.DataFrame, export: bool=True) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        This function returns dataframes containing important NDVI peak distributions over
        the years that are contained in the time-series data. This function can be used to
        aggregate the time-series data and make inferences about planting cycles and other
        important metrics.

        Args: (i) df_peaks - the NDVI dataframe with annotated peaks
              (ii) export - whether or not the results are to be exported; defaults to True

        Returns: (i) df_annual_peaks_monthly - occurrences of NDVI peaks every month over the years
                 (ii) df_annual_cum_peaks - distribution of numbers of NDVI peaks per farm over the years
        """
        # peak month and year are important for data aggregation
        if not pd.Series(["ndvi_peak_month", "ndvi_peak_year"]).isin(df_peaks.columns).all():
            df_peaks["ndvi_peak_month"] = df_peaks["ndvi_peak_date"].dt.month
            df_peaks["ndvi_peak_year"] = df_peaks["ndvi_peak_date"].dt.year

        # months through dt.month will be in the form of Int32
        if pd.api.types.is_integer_dtype(df_peaks["ndvi_peak_month"]):
            month_mapper = {
                1: "January",
                2: "February",
                3: "March",
                4: "April",
                5: "May",
                6: "June",
                7: "July",
                8: "August",
                9: "September",
                10: "October",
                11: "November",
                12: "December"
            }
            df_peaks["ndvi_peak_month"] = df_peaks["ndvi_peak_month"].map(month_mapper)

        # Number of monthly peaks over the years
        df_annual_peaks_monthly = df_peaks.groupby(["ndvi_peak_year", "ndvi_peak_month"]).agg(
            ndvi_peaks_per_month=pd.NamedAgg(column="ndvi_peak_date", aggfunc="count")
        ).reset_index()

        df_peaks_per_farm = df_peaks.groupby(["ndvi_peak_year", "uuid"]).agg(
            number_of_peaks_per_farm=pd.NamedAgg(column="ndvi_peak_date", aggfunc="count")
        ).reset_index()

        # Number of occurrences of peaks over the years
        df_annual_cum_peaks = df_peaks_per_farm.groupby(["ndvi_peak_year", "number_of_peaks_per_farm"]).agg(
            uuid_count=pd.NamedAgg(column="uuid", aggfunc="count")
        ).reset_index()

        if export:
            if not os.path.exists(f"vi_characteristics/{self.region}"):
                logging.info("Creating export directory.")
                os.makedirs(f"vi_characteristics/{self.region}", exist_ok=True)
            
            df_annual_peaks_monthly.to_csv(f"vi_characteristics/{self.region}/ndvi_peaks_monthly_{self.region}.csv", index=False)
            df_annual_cum_peaks.to_csv(f"vi_characteristics/{self.region}/ndvi_peaks_annual_{self.region}.csv", index=False)

            if not os.path.exists(f"plots/{self.region}"):
                os.makedirs(f"plots/{self.region}", exist_ok=True)

            # Export plots
            sns.barplot(df_annual_peaks_monthly, x="ndvi_peak_month", hue="ndvi_peak_year", edgecolor="k")
            plt.xlabel("Month", fontsize=12)
            plt.ylabel("Number of NDVI peaks", fontsize=12)
            plt.title("Number of NDVI peaks per month", fontsize=14)
            plt.savefig(f"plots/{self.region}/ndvi_peaks_monthly_{self.region}.png", dpi=300, bbox_inches="tight")

            sns.barplot(df_annual_cum_peaks, x="number_of_peaks_per_farm", hue="ndvi_peak_year", edgecolor="k")
            plt.xlabel("Number of annual NDVI peaks")
            plt.ylabel("Number of farms")
            plt.title("Distribution of number of annual NDVI peaks", fontsize=14)
            plt.savefig(f"plots/{self.region}/ndvi_peaks_annual_{self.region}.png", dpi=300, bbox_inches="tight")

        return df_annual_peaks_monthly, df_annual_cum_peaks
    
class ExtractNDMIData:
    def __init__(self, df: pd.DataFrame, region: str, validate_input: bool=True):
        # Check if input data has valid schema
        if validate_input:
            try:
                df = Schema(df)
            except SchemaError as e:
                logging.error("Data validation failed!")
                raise e
            
        self.df = df
        self.df_prepared = self._prepare_data() 
        self.region = region
    
    def _prepare_data(self) -> pd.DataFrame:
        df_copy = self.df.copy(deep=True)

        try:
            df_copy = df_copy[df_copy["polygon_type"]=="Farm"]
        except KeyError as e:
            logging.error("polygon_type column is not present in the dataframe.")
            raise e
        
        if not pd.api.types.is_datetime64_any_dtype(df_copy["date"]):
            df_copy["date"] = pd.to_datetime(df_copy["date"])

        return df_copy
    
    def high_ndmi_days(self, ndmi_threshold: float=0.38, export: bool=True) -> pd.DataFrame:
        df = self.df_prepared 

        # Filter out high-NDMI farms based on threshold
        df_high_ndmi = df[df["ndmi"] > ndmi_threshold]
        df_high_ndmi["year"] = df_high_ndmi["date"].dt.year

        """ 
        In order to be more precise about water-stress levels of farms, we
        take into account the (annual) cumulative number of days spent in
        high-NDMI zones.
        """
        df_high_ndmi_days = pd.DataFrame(
            df_high_ndmi.groupby(["uuid", "year"]).apply(
                lambda x: pd.Series({"high_ndmi_days": (x["date"].max() - x["date"].min()).days})
            )
        ).reset_index()
        
        if export:
            if not os.path.exists(f"vi_characteristics/{self.region}"):
                logging.info("Creating export directory.")
                os.makedirs(f"vi_characteristics/{self.region}", exist_ok=True)

            df_high_ndmi_days.to_csv(f"vi_characteristics/{self.region}/high_ndmi_days_{self.region}.csv", index=False)

            if not os.path.exists(f"plots/{self.region}"):
                os.makedirs(f"plots/{self.region}", exist_ok=True) 
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            sns.histplot(df_high_ndmi_days, y="high_ndmi_days", hue="year", multiple="stack", bins=40, palette="bright", ax=ax1)
            ax1.set_xlabel("Number of farms", fontsize=12)
            ax1.set_ylabel("Days", fontsize=12)

            sns.boxplot(df_high_ndmi_days, y="high_ndmi_days", hue="year", palette="bright", ax=ax2)
            ax2.set_xlabel("Year", fontsize=12)
            ax2.set_ylabel("Days", fontsize=12)

            fig.suptitle(r"Cumulative days with NDMI$\geq0.38$", fontsize=14)
            plt.savefig(f"plots/{self.region}/high_ndmi_days_{self.region}.png", dpi=300, bbox_inches="tight")

        return df_high_ndmi_days