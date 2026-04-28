"""
Classifier training node
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import Any

import mlflow
import pandas as pd
from sklearn.pipeline import Pipeline

from .cv_runner import run_stratified_cv
from .utils import (
    build_training_summary,
    fit_final_pipeline,
    log_training_run,
    select_model_params,
)

logger = logging.getLogger(__name__)

def _labels_for_panel_order(df_data: pd.DataFrame, df_label: pd.DataFrame) -> pd.Series:
    feature_uuid_order = pd.Index(df_data.index.get_level_values("uuid")).drop_duplicates()
    labels_by_uuid = (
        df_label.drop_duplicates(subset="uuid", keep="first")
        .set_index("uuid")["class_encoded"]
    )
    missing_labels = feature_uuid_order.difference(labels_by_uuid.index)

    if not missing_labels.empty:
        raise ValueError(
            "Training labels are missing UUIDs present in the feature panel: "
            f"{missing_labels.tolist()}"
        )

    return labels_by_uuid.loc[feature_uuid_order]


def train_classifier(
    df_data: pd.DataFrame,
    df_label: pd.DataFrame,
    training_params: dict[str, Any],
    feature_engineering_params: dict[str, Any],
) -> tuple[dict[str, Any], Pipeline]:
    """
    Classifier training node that receives inputs from `feature_engineering` pipeline.

    Args:
        df_data (pd.DataFrame): dataframe containing training data
        df_label (pd.DataFrame): dataframe containing training labels
        training_params (dict[str, Any]): the complete set of training parameters from
            `parameters.yml` under `training`.
        feature_engineering_params (dict[str, Any]): the complete set of feature engineering from
            `parameters.yml` under `feature_engineering.sklearn_transformation_pipeline`.
    """
    # Parse all training parameters
    model_name = training_params["active_model"]
    classifier_params = training_params["classifiers"][model_name]
    cv_params = training_params.get("cv", {})
    hpo_params = training_params.get("hyperparameter_search", {})
    mlflow_params = training_params.get("mlflow", {})
    target_label = int(training_params.get("target_label", 0))

    mlflow_enabled = bool(mlflow_params.get("enabled", True))
    tracking_uri = mlflow_params.get("tracking_uri", "file:./mlruns")
    experiment_name = mlflow_params.get("experiment_name", "timeseries_classification_local")

    if mlflow_enabled:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

    existing_run = mlflow.active_run()

    if mlflow_enabled and existing_run is None:
        run_context = mlflow.start_run(
            run_name=training_params.get("run_name", f"{model_name}-training")
        )
    else:
        run_context = nullcontext(existing_run)

    with run_context as active_run:
        # Best model parameters depending on whether or not optimization is required
        best_model_params = select_model_params(
            model_name=model_name,
            classifier_params=classifier_params,
            cv_params=cv_params,
            feature_engineering_params=feature_engineering_params,
            df_data=df_data,
            df_label=df_label,
            target_label=target_label,
            hpo_params=hpo_params,
            mlflow_enabled=mlflow_enabled,
        )

        cv_result = run_stratified_cv(
            model_name=model_name,
            df_data=df_data,
            df_label=df_label,
            model_params=best_model_params,
            fit_params=classifier_params.get("fit_params", {}),
            feature_engineering_params=feature_engineering_params,
            n_folds=int(cv_params.get("n_folds", 5)),
            random_state=int(cv_params.get("random_state", 42)),
            target_label=target_label,
        )

        summary = build_training_summary(
            model_name=model_name,
            df_label=df_label,
            cv_result=cv_result,
            best_model_params=best_model_params,
            tracking_uri=tracking_uri,
            experiment_name=experiment_name,
            run_id=active_run.info.run_id if active_run else None,
        )

        final_pipeline = fit_final_pipeline(
            model_name=model_name,
            df_data=df_data,
            labels=_labels_for_panel_order(df_data, df_label).to_numpy(),
            feature_engineering_params=feature_engineering_params,
            model_params=best_model_params,
            fit_params=classifier_params.get("fit_params", {}),
        )

        if mlflow_enabled:
            log_training_run(
                summary=summary,
                cv_result=cv_result,
                feature_engineering_params=feature_engineering_params,
                cv_params=cv_params,
                mlflow_params=mlflow_params,
                classifier_fit_params=classifier_params.get("fit_params", {}),
                final_pipeline=final_pipeline,
            )

        return summary, final_pipeline
