import mlflow 
#from mlflow.models import infer_signature
import pandas as pd
import numpy as np
from sktime.datatypes import check_is_scitype
from sklearn.base import BaseEstimator, TransformerMixin
from utils import RemoveNaNColumns
from sklearn.preprocessing import MinMaxScaler
from sktime.transformations.panel.catch22 import Catch22
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from plotting_utils import calculate_PR, plot_PR, plot_confusion_matrix
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import warnings
import logging
import optuna
from optuna.trial import Trial


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("feature_based_classifier.log"),
        logging.StreamHandler()
    ]
)
warnings.filterwarnings("ignore", category=RuntimeWarning)
optuna.logging.set_verbosity(optuna.logging.ERROR)

def transform_data(df: pd.DataFrame, df_label: pd.DataFrame) -> pd.DataFrame:
    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)

    # Labeled data does not contain two cases
    missing_uuid = list(set(df["uuid"].unique()) - set(df_label["uuid"].unique()))

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

def train_clf(
        df_transformed: pd.DataFrame,
        df_label: pd.DataFrame,
        scaler: TransformerMixin,
        clf: BaseEstimator
) -> tuple[float, float]:
    if "Unnamed: 0" in df_label.columns:
        df_label = df_label.drop(columns=["Unnamed: 0"])

    # Create integer categories if not present
    if "class_encoded" not in df_label.columns:
        df_label["class_encoded"] = df_label["class"].astype("category").cat.codes 
        
    uuids = df_label["uuid"].values
    labels = df_label["class_encoded"].values

    """ 
    Here, k-fold cross-validation is performed on the time-series data features
    extracted using Catch22. The transformations are chained using a `Pipeline`
    object to ensure that data leakage does not occur due to incorrect fits and
    transforms.
    """

    val_accuracy_scores = []
    val_f1_scores = []

    y_val_agg = []
    y_val_preds_agg = []

    # Initialize empty dictionaries for storing FPR, TPR and AUC
    n_classes = len(np.unique(labels))
    precision = {i: [] for i in range(n_classes)}
    recall = {i: [] for i in range(n_classes)}
    threshold = {i: [] for i in range(n_classes)}

    # Initialize confusion matrix for aggregation across folds
    CM_agg = np.zeros((n_classes, n_classes))

    SKF = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(SKF.split(uuids, labels)):
        logging.info(f"Stratified k-fold cross-validation, fold {fold + 1}")

        train_ids = uuids[train_idx]
        val_ids = uuids[val_idx]

        # Filtering the multi-index dataframe
        X_train = df_transformed.loc[
            df_transformed.index.get_level_values("uuid").isin(train_ids)
        ]

        X_val = df_transformed.loc[
            df_transformed.index.get_level_values("uuid").isin(val_ids)
        ]

        y_train = df_label.loc[df_label["uuid"].isin(train_ids), "class_encoded"]
        y_val = df_label.loc[df_label["uuid"].isin(val_ids), "class_encoded"]

        # Create a pipeline for performing Catch22 feature extraction, feature scaling and imputation
        # Implements a custom class for removing all columns containing all NaNs
        p = Pipeline(
            [
                ("catch22", Catch22(catch24=True)),
                ("scaler", scaler()),
                ("remove_nan", RemoveNaNColumns()),
                ("impute", SimpleImputer(strategy="mean"))
            ]
        )

        # Only apply transform() on validation data to prevent data leakage
        X_train_transformed = p.fit_transform(X_train)
        X_val_transformed = p.transform(X_val)

        # Obtain ROC curve and AUC score for each class for the fold
        # Update dictionaries
        precision_k, recall_k, threshold_k = calculate_PR(
            X_train_transformed, 
            y_train, 
            X_val_transformed, 
            y_val, 
            OneVsRestClassifier(clf)
        )

        # Append the scores for each class
        for i in range(n_classes):
            precision[i].append(precision_k[i])
            recall[i].append(recall_k[i])
            threshold[i].append(threshold_k[i])

        # Fitting the classifier to the training data and labels

        clf.fit(X_train_transformed, y_train)

        y_train_preds = clf.predict(X_train_transformed)
        y_val_preds = clf.predict(X_val_transformed)

        # Extend to aggregate list
        y_val_agg.extend(y_val)
        y_val_preds_agg.extend(y_val_preds)

        train_accuracy = accuracy_score(y_train, y_train_preds)
        logging.info(f"Training accuracy in fold {fold+1}: {train_accuracy}")
        val_accuracy = accuracy_score(y_val, y_val_preds)
        mlflow.log_metric(f"Validation accuracy in fold {fold+1}", val_accuracy)
        logging.info(f"Validation accuracy in fold {fold+1}: {val_accuracy}")

        # F1 score of class 0 (Farm)
        val_f1 = f1_score(y_val, y_val_preds, labels=[0], average="weighted")
        mlflow.log_metric(f"Validation F1 score of class 0 in fold {fold+1}", val_f1)
        logging.info(f"Validation F1 score of class 0 in fold {fold+1}: {val_f1}")

        val_accuracy_scores.append(val_accuracy)
        val_f1_scores.append(val_f1)

        # Compute confusion matrix for the fold
        CM = confusion_matrix(y_val, y_val_preds)
        CM_agg += CM

    #optimal_threshold = find_optimal_threshold(precision, recall, threshold)
    #mlflow.log_metric("Optimal threshold", optimal_threshold) 

    val_accuracy_scores_mean = np.mean(val_accuracy_scores)
    val_f1_scores_mean = np.mean(val_f1_scores)

    pr_plot = plot_PR(precision, recall, n_classes)
    mlflow.log_figure(pr_plot, "PR_curve.png")

    CM_plot = plot_confusion_matrix(CM_agg)
    mlflow.log_figure(CM_plot, "Confusion_matrix.png")

    clf_report = classification_report(y_val_agg, y_val_preds_agg, output_dict=True)
    mlflow.log_dict(clf_report, "classification_report.json")

    return val_accuracy_scores_mean, val_f1_scores_mean


def champion_callback(study, frozen_trial):
  """
  Logging callback that will report when a new trial iteration improves upon existing
  best trial values.

  Note: This callback is not intended for use in distributed computing systems such as Spark
  or Ray due to the micro-batch iterative implementation for distributing trials to a cluster's
  workers or agents.
  The race conditions with file system state management for distributed trials will render
  inconsistent values with this callback.
  """

  winner = study.user_attrs.get("winner", None)

  if study.best_value and winner != study.best_value:
      study.set_user_attr("winner", study.best_value)
      if winner:
          improvement_percent = (abs(winner - study.best_value) / study.best_value) * 100
          print(
              f"Trial {frozen_trial.number} achieved value: {frozen_trial.value} with "
              f"{improvement_percent: .4f}% improvement"
          )
      else:
          print(f"Initial trial {frozen_trial.number} achieved value: {frozen_trial.value}")

def objective(
        trial: Trial,
        df_transformed: pd.DataFrame,
        df_label: pd.DataFrame,
        scaler: TransformerMixin
    ) -> float:
    with mlflow.start_run(nested=True):
        lgbm_clf_params = {
            "objective": "multiclass", 
            "num_class": 3, 
            "metric": "multi_logloss",  
            "verbosity": -1,
            "boosting_type": "gbdt", 

            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 256),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 10.0, log=True),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),

            "boost_from_average": True,
        }

        model = LGBMClassifier(**lgbm_clf_params)
        val_accuracy_scores_mean, val_f1_scores_mean = train_clf(
            df_transformed,
            df_label,
            scaler,
            model
        )
        metrics = {
            "val_accuracy_mean": val_accuracy_scores_mean,
            "val_f1_mean": val_f1_scores_mean
        }
        
        mlflow.log_params(lgbm_clf_params)
        mlflow.log_metrics(metrics)

        return val_f1_scores_mean
    
def data_for_fitting(
        df_transformed: pd.DataFrame,
        df_label: pd.DataFrame,
        scaler: TransformerMixin
    ):
    """ 
    This function prepares data for final fitting for logged model.
    """
    if "Unnamed: 0" in df_label.columns:
        df_label = df_label.drop(columns=["Unnamed: 0"])

    # Create integer categories if not present
    if "class_encoded" not in df_label.columns:
        df_label["class_encoded"] = df_label["class"].astype("category").cat.codes 
    
    p = Pipeline(
            [
                ("catch22", Catch22(catch24=True)),
                ("scaler", scaler()),
                ("remove_nan", RemoveNaNColumns()),
                ("impute", SimpleImputer(strategy="mean"))
            ]
        )
    
    X_transformed = p.fit_transform(df_transformed)
    return X_transformed, df_label["class_encoded"].values

if __name__ == "__main__":
    DATA_PATH = "/Users/rafidmahbub/Desktop/DataKind_Geospatial/crop_classification/time_series_analyses/ndvi_series_labeled/ndvi_series_Trans_Nzoia_1_clean.csv"
    LABEL_PATH = "/Users/rafidmahbub/Desktop/DataKind_Geospatial/crop_classification/time_series_analyses/ndvi_series_labeled/ndvi_Trans_Nzoia_1_labels.csv"
    
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df_label = pd.read_csv(LABEL_PATH)

    df_transformed = transform_data(df, df_label)

    scaler = MinMaxScaler
    # mlflow tracking uri
    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

    mlflow.set_experiment("feature_based_classifier_baselines")
    with mlflow.start_run():
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: objective(trial, df_transformed, df_label, scaler),
            n_trials=50,
            callbacks=[champion_callback]
        )

        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_val_f1", study.best_value)

        model = LGBMClassifier(**study.best_params)

        X_final, y_final = data_for_fitting(df_transformed, df_label, scaler)

        model.fit(X_final, y_final)

        # #signature = infer_signature(df_transformed, RF_clf.predict(df_transformed))

        # # Log the model
        mlflow.lightgbm.log_model(
            lgb_model=model,
            artifact_path="timeseries_classification_LGBM",
            registered_model_name="timeseries_classification_LGBM",
            #signature=signature
        )