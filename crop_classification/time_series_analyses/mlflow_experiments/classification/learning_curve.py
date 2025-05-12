import numpy as np
import pandas as pd
from sktime.datatypes._panel._convert import from_multi_index_to_nested
from sktime.transformations.panel.catch22 import Catch22
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator
from sklearn.model_selection import learning_curve

def generate_learning_curve(
        df_transformed: pd.DataFrame,
        df_label: pd.DataFrame,
        model: BaseEstimator,
        model_params: dict[str, any] | None=None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ 
    This function generates the learning curve for a given model.

    Args: (i) df_transformed - NDVI time-series dataframe; must have two outer indices (uuid, date)
          (ii) df_label - an array of corresponding labels
          (iii) model - the classification model
          (iv) model_params - model hyperparameters

    Returns: (i) train_sizes - an array of training sizes used in the learning curve
             (ii) train_scores - an array of training scores
             (iii) val_scores - an array of validation scores
    """

    if "class_encoded" not in df_label.columns:
        # Encode labels if not already present
        df_label["class_encoded"] = df_label["class"].astype("category").cat.codes

    """ 
    Sklearn's learning_curve will not be able to process hierarchical time-series data. They have to
    be converted to nested arrays for the pipeline object to work with learning curve.
    """
    X = from_multi_index_to_nested(df_transformed)
    y = df_label["class_encoded"]

    # Provide all transformation steps
    if model_params is None:
        model_params = {}

    pipeline = make_pipeline(
        Catch22(catch24=True),
        SimpleImputer(strategy="mean"),
        MinMaxScaler(),
        model(**model_params)
    )

    train_sizes, train_scores, val_scores = learning_curve(
        pipeline,
        X,
        y, 
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1
    )

    return train_sizes, train_scores, val_scores

if __name__ == "__main__":

    from utils import transform_data
    from plotting_utils import plot_learning_curve
    from xgboost import XGBClassifier

    DATA_PATH = "/Users/rafidmahbub/Desktop/DataKind_Geospatial/crop_classification/time_series_analyses/tests/df_clean.csv"
    LABEL_PATH = "/Users/rafidmahbub/Desktop/DataKind_Geospatial/crop_classification/time_series_analyses/ndvi_series_labeled/Trans_Nzoia_1_tile_0_NDVI_labels.csv"
    
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df_label = pd.read_csv(LABEL_PATH)

    df_transformed = transform_data(df, df_label)
    model_params = {
        "n_estimators": 50,
        "gamma": 5,
        "lambda": 5,
        "alpha": 2.5,
        "max_depth": 5,
        "subsample": 0.75
    }

    train_sizes, train_scores, val_scores = generate_learning_curve(
        df_transformed,
        df_label,
        XGBClassifier
    )

    _ = plot_learning_curve(train_sizes, train_scores, val_scores, "XGBoost")
