import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sktime.datatypes import check_is_scitype
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

def transform_data(df: pd.DataFrame, df_label: pd.DataFrame) -> pd.DataFrame:
    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)

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

def train_test_split(
        df: pd.DataFrame,
        y: pd.Series,
        test_size: float=0.2,
        random_state: int=None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """ 
    This function generates train/test split on hierarchical NDVI time-series dataframe.

    Args: (i) df - NDVI time-series dataframe; must have two outer indices (uuid, date)
                   and must pass sktime scitype validation test
          (ii) y - an array of corresponding labels
    """
    if random_state is not None:
        np.random.seed(random_state)

    if "Unnamed: 0" in y.columns:
        y = y.drop(columns=["Unnamed: 0"])

    y = y.set_index("uuid")["class_encoded"]

    # Generate train/test split for uuid indices
    idx = df.index.get_level_values("uuid").unique()
    train_idx = np.random.choice(idx, size=int(len(idx) * (1 - test_size)), replace=False)
    test_idx = np.setdiff1d(idx, train_idx)

    df_train = df.loc[train_idx]
    df_test = df.loc[test_idx]

    y_train = y.loc[train_idx]
    y_test = y.loc[test_idx]

    return df_train, df_test, y_train, y_test

def find_optimal_threshold(
        precision: np.ndarray,
        recall: np.ndarray,
        threshold: np.ndarray,
        class_idx: int=0
) -> float:
    """ 
    This function returns the optimal threshold for the class which we are trying to
    optimize.

    Args: (i) precision - an array of precision values
          (ii) recall - an array of recall values
          (iii) theshold - an array of thresholds
    
    Returns: threshold_optimal - the optimal threshold
    """
    if len(precision[class_idx]) > 1:
        precision = np.mean(precision[class_idx], axis=0)

    if len(recall[class_idx]) > 1:    
        recall = np.mean(recall[class_idx], axis=0)

    if len(threshold[class_idx]) > 1:
        threshold = np.mean(threshold[class_idx], axis=0)

    f1 = np.nan_to_num(2*precision*recall / (precision + recall))

    max_f1_idx = np.argmax(f1)
    threshold_optimal = threshold[max_f1_idx]

    return threshold_optimal