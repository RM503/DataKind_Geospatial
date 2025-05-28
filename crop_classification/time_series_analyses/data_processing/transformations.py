import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sktime.datatypes import check_is_scitype
from sklearn.pipeline import Pipeline 
from sktime.transformations.panel.catch22 import Catch22
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

import logging

logging.basicConfig(level=logging.INFO)

class RemoveNaNColumns(TransformerMixin, BaseEstimator):
    """ 
    This class implements a transfomer inside an sklearn pipeline
    which finds all the columns where all elements are NaN and drops them.
    """
    def __init__(self):
        self.columns_to_drop = None
        
    def fit(self, X, y=None):
        # Validate data type
        X = self._validate_input(X)
        # Identify columns where all elements are NaN
        nan_cols = np.where(np.all(np.isnan(X), axis=0))[0]
   
        if len(nan_cols) > 0:
            logging.info("Removing columns with all NaN values")
        self.columns_to_drop = nan_cols 

        return self 
    
    def transform(self, X):
        # Drop columns
        X_transformed = X.copy()
        X_transformed = np.delete(X, self.columns_to_drop, axis=1)

        return X_transformed
    
    def _validate_input(self, X):
        if isinstance(X, np.ndarray):
            return X 
        else:
            raise TypeError("Object X does not have the required Numpy array format.")

def transform_data(df: pd.DataFrame, df_label: pd.DataFrame=None) -> pd.DataFrame:
    """ 
    This function transforms NDVI time-series data into hierarchically indexed
    sktime compatible panel data.

    Args: (i) df - NDVI time-series data to be transformed
          (ii) df_label - label data if used for training; defaults to None

    Returns: df_transformed - transformed dataframe following the sktime scitype
    """
    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)

    if df["date"].dtype != "datetime64[ns]":
        df["date"] = pd.to_datetime(df["date"])

    # Make sure date column in each uuid are monotonic increasing
    df = df.sort_values(by=["uuid", "date"], ascending=True).reset_index(drop=True)

    if df_label is not None:

        # If labeled data does not contain all polygons
        missing_uuid = np.setdiff1d(df["uuid"].unique(), df_label["uuid"].unique())

        if len(missing_uuid) > 0:
            df = df[~df["uuid"].isin(missing_uuid)]

    # Transform data into sktime scitype and check
    df_transformed = df.set_index(["uuid", "date"])[["ndvi"]]
    type_check = check_is_scitype(
        df_transformed,
        scitype="Panel",
        return_metadata=True
    )

    if type_check[0]:
        logging.info(f"Dataframe has correct 'scitype': {type_check[2]['scitype']}")
        return df_transformed
    else:
        logging.error(f"Dataframe does not have the correct 'scitype': {type_check[1]}")

def build_pipeline() -> Pipeline:
    """ 
    This function returns a pipeline object containing a sequence of
    transformations.
    """
    return Pipeline(
        [
            ("catch22", Catch22(catch24=True)),
            ("scaler", MinMaxScaler()),
            ("remove_nan", RemoveNaNColumns()),
            ("imputer", SimpleImputer(strategy="mean"))
        ]
    )
