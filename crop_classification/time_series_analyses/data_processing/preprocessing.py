""" 
Codes for data preprocessing and feature engineering.
"""
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from scipy.signal import savgol_filter
import pandera as pa
from pandera import DataFrameSchema, Column
from sklearn.ensemble import IsolationForest
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("preprocessing.log"),
        logging.StreamHandler()
    ]
)

class VIDataValidation:
    """ 
    This class validates the preprocessed time-series data containing any
    vegetation index useful for the study.
    """
    def __init__(self, vi_index: str):
        self.vi_index = vi_index
        self.schema = self._build_schema()

    def _build_schema(self) -> DataFrameSchema:
        # Returns the desired dataframe schema
        return DataFrameSchema(
            {
                "uuid": Column(str, nullable=False),
                "date": Column(pa.DateTime, nullable=False),
                self.vi_index: Column(float, checks=pa.Check.in_range(-1, 1), nullable=True)
            }
        )
    
    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.schema.validate(df)

def fill_dates(row: pd.Series) -> pd.Series:
    """ 
    This function implements a smart imputation strategy such that
    rows with missing entries at starting date is imputed with `bfill`
    while those with missing entries at the end are imputed with `ffill`.
    For others, it defaults to `bfill`. To be used if interpolation is not used.
    """
    if pd.isna(row.iloc[0]):
        row = row.bfill()
        return row
    elif pd.isna(row.iloc[-1]):
        row = row.ffill()
        return row
    else:
        row = row.bfill()
        return row
    
def find_outliers(col: pd.Series) -> NDArray[np.float64]:
    """ 
    This function applies the Isolation Forest algorith on the
    time-series data for detecting possible outliers.

    The value for contamination must account for the natural variation
    in the data. We choose 0.025.
    """

    X = col.values.reshape(-1, 1) # Format input into required shape
    model = IsolationForest(
        n_estimators=150,
        contamination=0.075,
        random_state=10
    ) # Setting contamination
    model.fit(X)

    # Predictions will consist of two values: +1 for inliers and -1 for outliers
    Y_preds = model.predict(X)

    return Y_preds

def date_resample(df: pd.DataFrame, vi: str) -> pd.DataFrame:
    """ 
    This function performs resampling on chunks of the dataframe (based on uuid)
    to remove irregular time samples by resample to 5 day intervals and interpolating
    the additional fields.

    Resampling is recommended if there no large gaps between extracted data. There will
    be instances where this breaks.
    """
    if df["date"].dtype != "datetime64[ns]":
        # Convert date column to datetime if not already 
        df["date"] = pd.to_datetime(df["date"], coerce=True)

    if len(df["date"].diff().value_counts()) > 1:
        # If there are multiple `periods` in the data

        df = (
            df.set_index("date").resample("5D")
              .asfreq()
        )

        df[vi] = df[vi].interpolate()
        df["uuid"] = df["uuid"].fillna(df["uuid"].mode()[0])

        return df.reset_index()
    else:
        return df

def clean_vi_series(
        df: pd.DataFrame,
        vi: str,
        fill_method: str="interpolate",
        date_resample: bool=True
    ) -> pd.DataFrame:
    """ 
    Restructures the VI row-major table by melting the dataframe, in effect,
    stack time-series for each uuid vertically.
    """
    df = df.copy()
    if pd.Series(["system:index", ".geo"]).isin(df.columns).all():
        logging.info("Getting rid of useless columns")
        df = df.drop(columns=["system:index", ".geo"]) # Remove useless columns

    if "uuid" in df.columns:
        new_cols = ["uuid"] + [col for col in df.columns if col != "uuid"]
        df = df.reindex(columns=new_cols)
    else:
        logging.error("No uuid column found")

    # Isolate only the numerical portion of the dataframe
    if fill_method == "interpolate":
        
        df.iloc[:, 1:] = df.iloc[:, 1:].interpolate(method="linear", axis=1)

    elif fill_method == "simple":
        
        df.iloc[:, 1:] = df.iloc[:, 1:].apply(fill_dates, axis=1)

    
    df_melted = (
    df.melt(id_vars="uuid", var_name="date", value_name=vi)
    )
    df_melted["date"] = pd.to_datetime(df_melted["date"])

    df_melted = (
    df_melted.drop_duplicates(subset=["uuid", "date"], keep="first")
             .sort_values(by=["uuid", "date"])
             .reset_index(drop=True)
    )   

    # Applying Savitzky-Golay filter for smoothing time-series data and resample time
    WINDOW_SIZE = 7
    POLY_ORDER = 3

    groups = []
    for _, group in df_melted.groupby("uuid"):
        group[vi] = savgol_filter(group[vi], WINDOW_SIZE, POLY_ORDER)

        if date_resample:
            groups.append(date_resample(group, vi))
        else:
            groups.append(group)

    df_smoothed = pd.concat(groups).reset_index(drop=True)

    """ 
    The outliers are tagged applying the `find_outliers` function to the `vi` column
    in uuid groups. Since we expect the outliers to be incorrect calculations arising
    from GEE aggregation, we should not remove them. Instead, they are set to the value
    at the previous date. 
    """


    df_smoothed["outlier"] = df_smoothed.groupby("uuid")[vi].transform(find_outliers)

    # Set outliers to NaN and then fill them using `bfill`
    #condition = df_melted["outlier"] == -1
    df_smoothed.loc[df_smoothed["outlier"] == -1, vi] = np.nan

    df_clean = (
        df_smoothed.bfill()
                .drop(columns="outlier")
    )
 
    # Even if there are negative VI after outlier removal, set them to 0
    df_clean.loc[
        df_clean[vi] < -1.0, vi
    ] = -1.0
    # Even if there are NDVI values over 1.0 after outlier removal, set them to 1.0
    df_clean.loc[
        df_clean[vi] > 1.0, vi
    ] = 1.0

    # Check if cleaned data conforms to required schema
    try:
        validator = VIDataValidation(vi)
        validator.validate(df_clean)
        logging.info("Data validation passed")

        return df_clean
    except pa.errors.SchemaErrors as e:
        logging.error(f"Data validation failed: {e}")

if __name__ == "__main__":
    # Test
    FILE_PATH = "/Users/rafidmahbub/Desktop/DataKind_Geospatial/crop_classification/time_series_analyses/ndvi_series_raw/ndvi_series_Kajiado_1_tile_0.csv"
    df = pd.read_csv(FILE_PATH)

    df_cleaned = clean_vi_series(df, "ndvi", date_resample=False)

    # OUTPUT_NAME = f"{FILE_PATH.split('/')[-1].split('.')[0]}_clean.csv"
    # df_cleaned.to_csv(OUTPUT_NAME, index=False)
    print(df_cleaned)