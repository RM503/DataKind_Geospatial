""" 
Codes for data preprocessing and feature engineering.
"""
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import pandera as pa
from pandera import (Field, DataFrameModel)
from pandera.typing import Series
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

class ndvi_data_validation(DataFrameModel):
    """
    This class implements a pandera schema for data validation.
    """
    uuid: Series[str] = Field(nullable=False)
    date: Series[pa.DateTime] = Field(nullable=False)
    ndvi: Series[float] = Field(nullable=True, ge=0, le=1) # NDVI ranges from 0 to 1

def fill_dates(row: pd.Series) -> pd.Series:
    """ 
    This function implements a smart imputation strategy such that
    rows with missing entries at starting date is imputed with `bfill`
    while those with missing entries at the end are imputed with `ffill`.
    For others, it defaults to `bfill`.
    """
    if pd.isna(row.iloc[0]):
        row = row.bill()
        return row
    elif pd.isna(row.iloc[-1]):
        row = row.ffill()
        return row
    else:
        row = row.bfill()
        return row
    
def find_outliers(col: pd.Series) -> np.ndarray:
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

def date_resample(df: pd.DataFrame) -> pd.DataFrame:
    """ 
    This function performs resampling on chunks of the dataframe (based on uuid)
    to remove irregular time samples by resample to 5 day intervals and interpolating
    the additional fields.
    """
    if len(df["date"].diff().value_counts()) > 1:
        # If there are multiple `periods` in the data
        logging.info(f"Resampling {df['uuid'].iloc[0]}")

        df = (
            df.set_index("date").resample("5D")
              .asfreq()
              .ffill()
        )

        return df.reset_index()
    else:
        return df

def clean_ndvi_series(df: pd.DataFrame) -> pd.DataFrame:
    """ 
    Restructures the NDVI row-major table by melting the dataframe, in effect,
    stack time-series for each uuid vertically.
    """
    if pd.Series(["system:index", ".geo"]).isin(df.columns).all():
        logging.info("Getting rid of useless columns")
        df = df.drop(columns=["system:index", ".geo"]) # Remove useless columns

    uuid_col = df.columns[-1]
    new_cols = [uuid_col] + list(df.columns[:-1])
    df = df.reindex(columns=new_cols)

    # Isolate only the numerical portion of the dataframe
    df.iloc[:, 1:] = df.iloc[:, 1:].apply(fill_dates, axis=1)

    df_melted = (
        df.melt(id_vars="uuid", var_name="date", value_name="ndvi")
        .groupby("uuid", group_keys=False).apply(lambda row: row.sort_values(by="date", ascending=False))
    )
    df_melted["date"] = pd.to_datetime(df_melted["date"])
    df_melted = df_melted.drop_duplicates(subset=["uuid", "date"], keep="first").reset_index(drop=True)
    df_melted["ndvi"] = (
        df_melted["ndvi"].apply(lambda x: 0 if x < 0 else x)
                         .astype(float)
    )

    # Applying Savitzky-Golay filter for smoothing time-series data and resample time
    WINDOW_SIZE = 7
    POLY_ORDER = 3

    groups = []
    for _, group in df_melted.groupby("uuid"):
        group["ndvi"] = savgol_filter(group["ndvi"], WINDOW_SIZE, POLY_ORDER)
        groups.append(date_resample(group))

    df_smoothed = pd.concat(groups).reset_index(drop=True)

    """ 
    The outliers are tagged applying the `find_outliers` function to the `ndvi` column
    in uuid groups. Since we expect the outliers to be incorrect calculations arising
    from GEE aggregation, we should not remove them. Instead, they are set to the value
    at the previous date. 
    """

    df_smoothed["outlier"] = df_smoothed.groupby("uuid")["ndvi"].transform(find_outliers)

    # Set outliers to NaN and then fill them using `bfill`
    #condition = df_melted["outlier"] == -1
    df_smoothed.loc[df_melted["outlier"] == -1, "ndvi"] = np.nan

    df_clean = (
        df_smoothed.bfill()
                 .drop(columns="outlier")
    )

    # Check if cleaned data conforms to required schema
    try:
        ndvi_data_validation.validate(df_clean, lazy=True) # Allow lazy evaluation
        logging.info("Data validation passed")

        return df_clean
    except pa.errors.SchemaErrors as e:
        logging.error(f"Data validation failed: {e}")

if __name__ == "__main__":
    # Test
    df = pd.read_csv("/Users/rafidmahbub/Desktop/DataKind_Geospatial/crop_classification/time_series_analyses/ndvi_series/ndvi_series_Trans_Nzoia_1_tile_0.csv")

    df_cleaned = clean_ndvi_series(df)
    df_cleaned.to_csv("df_clean.csv", index=False)