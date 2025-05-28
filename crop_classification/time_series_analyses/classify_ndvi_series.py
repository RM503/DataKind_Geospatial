""" 
Codes for performing sample inference and large scale predictions
using trained XGBoost classification models.
"""
import os
import mlflow 
from mlflow.tracking import MlflowClient
from param_type_caster import ParamTypeCaster
import data_processing.transformations as transformations
from generate_ndvi_clean import get_file_paths
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from xgboost import XGBClassifier
import logging
from tqdm import tqdm
import warnings

logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore", category=RuntimeWarning)

class NDVIClassifier:
    def __init__(self, file_paths: list[str], clf: XGBClassifier):
        if not isinstance(file_paths, list):
            logging.warning(f"{file_paths} should be a list of file paths")

        # Checks if XGBoost has been fitted
        # if not hasattr(clf, "booster_"):
        #     logging.warning("Classifier has not been fitted")

        self.file_paths = file_paths
        self.pipeline = transformations.build_pipeline()
        self.clf = clf

    def separate_by_region(self) -> list[list[str]]:
        """
        This function separates the list of time-series file_paths
        by region for better aggregation.
        """ 
        regions = [
            "Kajiado_1",
            "Kajiado_2",
            "Laikipia_1",
            "Machakos_1",
            "Mashuru_1",
            "Trans_Nzoia_1"
        ]
        # For storing file paths by region as nested lists
        # nested_file_paths can have empty sublists if no file found with matches
        nested_file_paths = []
        for region in regions:
            file_paths_by_region = [path for path in self.file_paths if region in path]
            nested_file_paths.append(file_paths_by_region)

        return nested_file_paths

    def prepare_data(self, file_path: str) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """ 
        This function applies the necessary transformations on a single dataframe
        to be used for inference/prediction.

        Args: file_path - the file path of the NDVI time-series data
        
        Returns: the transformed data in the form an an ndarray
        """
        if not os.path.exists(file_path):
            logging.error(f"File in {file_path} cannot be found.")

        df = pd.read_csv(file_path)
        df_transformed = transformations.transform_data(df)

        uuid_list = np.unique(df["uuid"])

        # Fit and transform the time-series data on the pipeline
        X_transformed = self.pipeline.fit_transform(df_transformed)

        return X_transformed, uuid_list

    def sample_inference(
        self,
        sample_file_path: str=None, 
        num: int=10, 
        export: bool=True
    ) -> pd.DataFrame:
        """ 
        This function performs inference on a single file for testing purposes.

        Args: (i) sample_file_path - path to NDVI time-series file for testing; defaults self.file_paths[0]
                                     if None
              (ii) num - number of randomized entries to consider in inference; defaults to 10
              (iii) export - whether inference data should be exported; defaults to True
        """
        if sample_file_path is None:
            sample_file_path = self.file_paths[0]

        X, uuid_list = self.prepare_data(sample_file_path)

        """
        For quick inference and visual inspection, we select a random subset of 
        uuid values.
        """
        # uuid_list_random = np.random.choice(uuid_list, size=num, replace=False)
        # idx_random = np.intersect1d(uuid_list_random, uuid_list)[-1]
        # X_random = X[idx_random, :]
        idx_random = np.random.choice(len(uuid_list), size=num, replace=False)
        X_random = X[idx_random, :] 
        uuid_list_random = uuid_list[idx_random]

        y_preds = self.clf.predict(X_random)
        predictions = pd.DataFrame(
            {
                "uuid": uuid_list_random,
                "prediction": y_preds
            }
        )

        if export:
            export_file_name = f"{sample_file_path.split('/')[-1].split('.')[0]}_predictions.csv"
            predictions.to_csv(f"inference/test/{export_file_name}", index=False)

        return predictions

    def make_predictions(self):
        """
        This function performs inference on a large collection of data...
        """
        file_paths_by_region = self.separate_by_region() 

        for path_list in file_paths_by_region:
            if len(path_list) > 0:
                # Ignore empty nested lists
                df_preds_region = []
                for _, path in tqdm(enumerate(path_list)):
                    logging.info(f"Working on {path}")
                    X, uuid_list = self.prepare_data(path)
                    
                    """ 
                    The classifier was trained on 23 features. This is because one of the
                    features in the training data was consistently NaN for all instances.
                    However, there might be instances where this feature is not always NaN.
                    In such cases, the classifier will not be able to perform inference. 
                    In such cases, we consciously remove that column after pipeline transformation.
                    """
                    if X.shape[1] != 23:
                        # Remove the 4th index column since that is the one that is predominantly all NaN
                        X = np.delete(X, 4, axis=1)

                    y_preds = self.clf.predict(X)
                    predictions = pd.DataFrame(
                        {
                            "uuid": uuid_list,
                            "prediction": y_preds
                        }
                    )

                    df_preds_region.append(predictions)

                # All predictions from each concatenated
                df_preds_region_concat = pd.concat(df_preds_region).reset_index(drop=True)
                label_map = {
                    0: "Farm",
                    1: "Field",
                    2: "Other",
                    3: "Tree"
                }

                # Apply a label_map for better interpretability
                df_preds_region_concat["prediction_decoded"] = df_preds_region_concat["prediction"].map(label_map)

                export_file_name = f"{path.split('/')[-1].split('_tile')[0]}_predictions.csv"
                df_preds_region_concat.to_csv(f"inference/{export_file_name}", index=False)

if __name__ == "__main__":
    # Include tracking server information
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    model_uri = 'runs:/d2401c44ac354a3bb386f1051c6cd247/timeseries_classification_XGB'
    run_id = model_uri.split("/")[1]

    # Initialize mlflow client
    client = MlflowClient()
    run = client.get_run(run_id)

    """ 
    MLFlow logs all parameter key-values as strings by default. These have to be
    cast back to their original forms for predictions to work.
    """
    params_logged = run.data.params

    # Initialize param type caster
    param_schema_path = "param_schemas/xgboost_schema.json"
    caster = ParamTypeCaster()
    caster.load_schema(param_schema_path)

    params = caster.cast_params(params_logged)

    # Load trained xgboost model with optimal parameters
    model_path = "../../mlartifacts/449520651186519710/d2401c44ac354a3bb386f1051c6cd247/artifacts/timeseries_classification_XGB/model.xgb"
    xgb_clf = XGBClassifier(**params)
    xgb_clf.load_model(model_path)

    #################################
    regions = ["Kajiado_2", "Machakos_1", "Mashuru_1"]
    file_paths = get_file_paths(regions, input_dir="ndvi_series_clean")

    ndvi_clf = NDVIClassifier(file_paths, xgb_clf)
    
    predictions = ndvi_clf.make_predictions()