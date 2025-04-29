import mlflow
from mlflow.models import infer_signature
import numpy as np
import pandas as pd
from sktime.datatypes import check_is_scitype
from sktime.dists_kernels import DtwDist
from umap import UMAP
from sktime.clustering.k_means import TimeSeriesKMeans
from sktime.utils import mlflow_sktime

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("clustering.log"),
        logging.StreamHandler()
    ]
)

def generate_random_sample(df: pd.DataFrame, n: int=500) -> pd.DataFrame:
    """ 
    This function generates a random sample of the NDVI time-series data for
    easier clustering experimentation.

    Args: (i) df - NDVI time-series data in long format; gets modified into the appropriate format
          (ii) n - the number of polygon uuid to consider in the sample; defaults to 500

    Returns: the random sample modified to the appropriate format
    """
    np.random.seed(11)
    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)

    df["date"] = pd.to_datetime(df["date"])

    uuid_list = df["uuid"].unique() # an array of unique uuid to sample from
    if n == uuid_list.shape[0]:
        logging.warning(f"Sample size {n} is equal to dataframe uuid size")
    if n > uuid_list.shape[0]:
        logging.error(f"Sample size {n} is larger than dataframe uuid size")

    uuid_random = np.random.choice(uuid_list, n, replace=False)
    df_random = df[df["uuid"].isin(uuid_random)]

    """ 
    sktime works with hierarchical dataframes for analysis large numbers of time-series data.
    Now, the df_random is transformed in one with an outermost uuid level, following by date
    and then ndvi.
    """
    df_transformed = df_random.set_index(["uuid", "date"])[["ndvi"]] # set hierarchical indexing

    type_check = check_is_scitype(
        df_transformed,
        scitype="Panel",
        return_metadata=True
    )

    if not type_check[0]:
        logging.error(f"Dataframe has incorrect 'scitype': {type_check[1]}")

    return df_transformed

def dtw_matrix(df: pd.DataFrame) -> np.ndarray:
    return DtwDist().transform(df)

if __name__ == "__main__":
    # mlflow tracking uri
    mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

    # Create a new MLflow Experiment
    mlflow.set_experiment("kmeans_clustering_timeseries")
    with mlflow.start_run():
        FILE_PATH = "/Users/rafidmahbub/Desktop/DataKind_Geospatial/crop_classification/time_series_analyses/tests/df_clean.csv"

        df = pd.read_csv(FILE_PATH)

        df_random_sample = generate_random_sample(df)
        dtw_matrix = dtw_matrix(df_random_sample)

        # Fitting UMAP to a precomputed DTW matrix is computationally efficient
        umap_model = UMAP(
            n_neighbors=12,
            n_components=3,
            metric="precomputed",
            random_state=10
        )
        umap_embedding = umap_model.fit_transform(dtw_matrix)

        kmeans_params = {
            "n_clusters": 6,
            "init_algorithm": "kmeans++",
            "metric": "dtw",
            "random_state": 10
        }
        mlflow.log_params(kmeans_params)

        kmeans_model = TimeSeriesKMeans(**kmeans_params)
        kmeans_model.fit(df_random_sample)

        labels = kmeans_model.labels_
        centroids = kmeans_model.cluster_centers_

        signature = infer_signature(df_random_sample, labels)

        # Log the model
        model_info = mlflow_sktime.log_model(
            sktime_model=kmeans_model,
            artifact_path="kmeans_timeseries",
            registered_model_name="kmeans_model_timeseries",
            signature=signature
        )