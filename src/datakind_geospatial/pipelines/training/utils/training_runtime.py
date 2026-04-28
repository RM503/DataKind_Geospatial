from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import Any, Literal, TypedDict

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import (
    PrecisionRecallDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize

from datakind_geospatial.pipelines.feature_engineering.nodes import build_sklearn_pipeline

from ..cv_runner import run_stratified_cv
from ..models.registry import get_classifier

logger = logging.getLogger(__name__)


class SearchParamSpec(TypedDict, total=False):
    type: Literal["categorical", "float", "int"]
    low: float | int
    hight: float | int
    log: bool


SearchSpace = dict[str, SearchParamSpec]

class ModelHPOConfig(TypedDict):
    search_space: SearchSpace


class HPOConfig(TypedDict):
    enabled: bool
    direction: Literal["maxmimize", "minimize"]
    n_trials: int
    model: dict[str, ModelHPOConfig]


def select_model_params(
    *,
    model_name: str,
    classifier_params: dict[str, Any],
    cv_params: dict[str, Any],
    feature_engineering_params: dict[str, Any],
    df_data: pd.DataFrame,
    df_label: pd.DataFrame,
    target_label: int,
    hpo_params: HPOConfig,
    mlflow_enabled: bool,
) -> dict[str, Any]:
    """
    Performs Optuna hyperparameter optimization to return best model parameters if enable, else
    returns base model parameters.

    Args:
        model_name (str): Name of the classifier model
        classifier_params (dict[str, Any]): Base classifier model parameters
        cv_params (dict[str, Any]): Cross-validation parameters
        feature_engineering_params (dict[str, Any]): Cross-validation aware feature engineering parameters
        df_data (pd.DataFrame): Training dataframe
        df_label (pd.DataFrame): Label dataframe
        target_label (int): Target class in classification training
        hpo_parameters (HPOConfig): Hyperparameter-optimization configuration
        mlflow_enabled (bool): Flag for whether or not MLFlow is enabled

    Returns:
        dict[str, Any]: Best parameters for the model (base or optimized)
    """
    base_model_params = classifier_params.get("model_params", {}).copy()

    # User base parameters if hyperparameter optimization is disabled
    if not hpo_params.get("enabled", False):
        return base_model_params

    search_space = hpo_params["model"][model_name].get("search_space", {})
    n_trials = int(hpo_params.get("n_trials", 10))
    direction = hpo_params.get("direction", "maximize")

    def objective(trial: optuna.Trial) -> float:
        # Override base_model_params with suggest_trial_params
        trial_params = base_model_params | suggest_trial_params(trial, search_space)
        nested_run = mlflow.start_run(
            nested=True,
            run_name=f"{model_name}-trial-{trial.number}"
        ) if mlflow_enabled else nullcontext()

        with nested_run:
            cv_result = run_stratified_cv(
                model_name=model_name,
                df_data=df_data,
                df_label=df_label,
                model_params=trial_params,
                fit_params=classifier_params.get("fit_params", {}),
                feature_engineering_params=feature_engineering_params,
                n_folds=int(cv_params.get("n_folds", 5)),
                random_state=int(cv_params.get("random_state", 42)),
                target_label=target_label,
            )
            selection_metric = float(cv_result["selection_metric"])

            if mlflow_enabled:
                mlflow.log_params({f"trial.{key}": value for key, value in trial_params.items()})
                mlflow.log_metric("trial.selection_metric", selection_metric)

            return selection_metric

    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials)

    return base_model_params | study.best_params


def suggest_trial_params(
    trial: optuna.Trial,
    search_space: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """
    Parses hyperparameter space for a given model name. In `conf/base/parameters.yml,
    this falls under training.hyperparameter_search.model.[model_name].search_space.

    Args:
        trial (optuna.Trial): The Optuna trial
        search_space (dict[str, Any]): The parameter space dict for the give classifier

    Returns:
        (dict[str, Any]): A parsed version of parameter space for Optuna.
    """
    suggested_params: dict[str, Any] = {}

    for name, config in search_space.items():
        param_type = config["type"]
        if param_type == "float":
            suggested_params[name] = trial.suggest_float(
                name,
                float(config["low"]),
                float(config["high"]),
                log=bool(config.get("log", False)),
            )
        elif param_type == "int":
            suggested_params[name] = trial.suggest_int(
                name,
                int(config["low"]),
                int(config["high"]),
                log=bool(config.get("log", False)),
            )
        elif param_type == "categorical":
            suggested_params[name] = trial.suggest_categorical(name, config["choices"])
        else:
            raise ValueError(f"Unsupported Optuna parameter type '{param_type}' for '{name}'.")

    return suggested_params


def build_training_summary(
    *,
    model_name: str,
    df_label: pd.DataFrame,
    cv_result: dict[str, Any],
    best_model_params: dict[str, Any],
    tracking_uri: str,
    experiment_name: str,
    run_id: str | None,
) -> dict[str, Any]:
    """
    Builds a training summary after model training.

    Args:
        model_name (str): Name of the classifier model
        df_label (pd.DataFrame): Label dataframe
        cv_result (dict[str, Any]): Results generated from cross-validated training
        best_model_params (dict[str, Any]): Best parameters for the model for current run
        tracking_uri (str): MLFlow tracking server URI
        experiment_name (str): MLFlow experiment name
        run_id (str): Identifier for current run

    Returns:
        (dict[str, Any]): Detailed summary of training run
    """
    y_true = np.asarray(cv_result["labels"])
    y_pred = np.asarray(cv_result["predictions"])
    y_prob = np.asarray(cv_result["probabilities"])
    classes = [int(class_label) for class_label in cv_result["classes"]]

    summary_metrics = {
        "validation_accuracy_mean": float(accuracy_score(y_true, y_pred)),
        "validation_macro_f1_mean": float(f1_score(y_true, y_pred, average="macro")),
        "validation_target_f1_mean": float(cv_result["selection_metric"])
    }

    class_name_map = (
        df_label[["class_encoded", "class"]]
        .drop_duplicates()
        .sort_values("class_encoded")
    )

    return {
        "model_name": model_name,
        "best_model_params": best_model_params,
        "fold_metrics": cv_result["fold_metrics"],
        "summary_metrics": summary_metrics,
        "classes": classes,
        "class_names": class_name_map["class"].tolist(),
        "classification_report": classification_report(y_true, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=classes).tolist(),
        "oof_probabilities_shape": list(y_prob.shape),
        "mlflow": {
            "tracking_uri": tracking_uri,
            "experiment_name": experiment_name,
            "run_id": run_id,
        },
    }


def fit_final_pipeline(
    *,
    model_name: str,
    df_data: pd.DataFrame,
    labels: np.ndarray,
    feature_engineering_params: dict[str, Any],
    model_params: dict[str, Any],
    fit_params: dict[str, Any],
) -> Pipeline:
    """
    Creates a final fitted feature engineering + model pipeline on the full training data. 
    This is used for MLFlow model logging and for generating final model artifacts.

    Args:
        model_name (str): Name of the classifier model
        df_data (pd.DataFrame): Training dataframe
        labels (np.ndarray): Training labels
        feature_engineering_params (dict[str, Any]): Parameters for feature engineering
        model_params (dict[str, Any]): Parameters for the classifier model
        fit_params (dict[str, Any]): Parameters for fitting the model

    Returns:
        (Pipeline): Fitted pipeline
    """
    classifier_spec = get_classifier(model_name)
    feature_pipeline = build_sklearn_pipeline(feature_engineering_params)
    classifier = classifier_spec.build_model(model_params)
    training_pipeline = Pipeline(
        [
            ("feature_extractor", feature_pipeline),
            ("classifier", classifier),
        ]
    )

    pipeline_fit_params = {
        f"classifier__{param_name}": param_value
        for param_name, param_value in fit_params.items()
        if param_name != "eval_set"
    }
    training_pipeline.fit(df_data, labels, **pipeline_fit_params)
    return training_pipeline


def log_training_run(
    *,
    summary: dict[str, Any],
    cv_result: dict[str, Any],
    feature_engineering_params: dict[str, Any],
    cv_params: dict[str, Any],
    mlflow_params: dict[str, Any],
    classifier_fit_params: dict[str, Any],
    final_pipeline: Pipeline,
) -> None:
    """
    Logs training run results to MLFlow, including parameters, metrics, artifacts
    (confusion matrix and precision-recall curves), and the final model.

    Args:
        summary (dict[str, Any]): Training summary generated from `build_training_summary`
        cv_result (dict[str, Any]): Results generated from cross-validated training
        feature_engineering_params (dict[str, Any]): Parameters for feature engineering
        cv_params (dict[str, Any]): Parameters for cross-validation
        mlflow_params (dict[str, Any]): Parameters for MLFlow logging
        classifier_fit_params (dict[str, Any]): Parameters for fitting the classifier model
        final_pipeline (Pipeline): Final fitted pipeline on the full training data

    Returns:
        None
    """
    flattened_params = {
        "model_name": summary["model_name"],
        **{f"feature_engineering.{key}": value for key, value in feature_engineering_params.items()},
        **{f"cv.{key}": value for key, value in cv_params.items()},
        **{f"model.{key}": value for key, value in summary["best_model_params"].items()},
        **{f"fit.{key}": value for key, value in classifier_fit_params.items()},
    }
    mlflow.log_params(flattened_params)
    mlflow.log_metrics(summary["summary_metrics"])
    mlflow.log_dict(summary, "training_summary.json")

    confusion = np.asarray(summary["confusion_matrix"])
    confusion_figure = plot_confusion_matrix(
        confusion,
        [str(class_name) for class_name in summary["class_names"]],
    )
    mlflow.log_figure(confusion_figure, "confusion_matrix.png")
    plt.close(confusion_figure)

    pr_figure = plot_precision_recall_curves(
        y_true=np.asarray(cv_result["labels"]),
        y_prob=np.asarray(cv_result["probabilities"]),
        labels=np.asarray(summary["classes"]),
        class_names=[str(class_name) for class_name in summary["class_names"]],
    )
    if pr_figure is not None:
        mlflow.log_figure(pr_figure, "precision_recall_curves.png")
        plt.close(pr_figure)

    if mlflow_params.get("log_model", True):
        log_model_kwargs = {
            "sk_model": final_pipeline,
            "artifact_path": mlflow_params.get("artifact_path", "timeseries_classifier"),
        }
        registered_model_name = mlflow_params.get("registered_model_name")
        if registered_model_name:
            log_model_kwargs["registered_model_name"] = registered_model_name
        mlflow.sklearn.log_model(**log_model_kwargs)


def plot_confusion_matrix(confusion: np.ndarray, class_names: list[str]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(confusion, cmap="Blues")
    ax.figure.colorbar(image, ax=ax)
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Cross-Validated Confusion Matrix")

    for row_idx in range(confusion.shape[0]):
        for col_idx in range(confusion.shape[1]):
            ax.text(col_idx, row_idx, int(confusion[row_idx, col_idx]), ha="center", va="center")

    fig.tight_layout()
    return fig


def plot_precision_recall_curves(
    *,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
) -> plt.Figure | None:
    if y_prob.size == 0:
        return None

    y_true_bin = label_binarize(y_true, classes=labels)
    if y_true_bin.ndim == 1:
        y_true_bin = y_true_bin.reshape(-1, 1)
    fig, ax = plt.subplots(figsize=(8, 6))

    n_curves = min(y_true_bin.shape[1], y_prob.shape[1], len(labels))
    for idx in range(n_curves):
        label = labels[idx]
        display = PrecisionRecallDisplay.from_predictions(
            y_true_bin[:, idx],
            y_prob[:, idx],
            name=class_names[idx] if idx < len(class_names) else f"class_{label}",
            ax=ax,
        )
        display.line_.set_linewidth(2)

    ax.set_title("Cross-Validated Precision-Recall Curves")
    fig.tight_layout()
    return fig
