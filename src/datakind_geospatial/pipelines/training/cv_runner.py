"""
Module containing runner for cross-validation training loop.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

from datakind_geospatial.pipelines.feature_engineering.nodes import build_sklearn_pipeline

from .models.base import fit_one_fold
from .models.registry import get_classifier

logger = logging.getLogger(__name__)

def run_stratified_cv(
    model_name: str,
    df_data: pd.DataFrame,
    df_label: pd.DataFrame,
    model_params: dict[str, Any],
    fit_params: dict[str, Any],
    feature_engineering_params: dict[str, Any],
    n_folds: int,
    random_state: int,
    target_label: int = 0,
) -> dict[str, Any]:
    """
    Implememnts a runner for performing k-fold stratified cross-validation. This
    function is invoked on the training node of the Kedro pipeline.

    Args:
        model_name (str): Name of a supported classification model.
        df_data (pd.DataFrame): re-indexed training data passed from `feature_engineering` pipeline.
        df_label (pd.DataFrame): encoded training labels passed from `feature_engineering` pipeline.
        model_params (dict[str, Any]): dictionary of model parameters.
        fit_params (dict[str, Any]): dictionary of model parameters.
        feature_engineering_params (dict[str, Any]): dictionary of feature engineering parameters.
        n_folds (int): number of folds to use.
        random_state (int): random state to use.
        target_label (int): target label to use.

    Returns:
        dict[str, Any]: dictionary of training outcomes and metrics.
    """
    classifier_spec = get_classifier(model_name)

    # Generate arrays containing uuids and labels for uuid-aware splits
    uuids = df_label["uuid"].to_numpy()
    labels = df_label["class_encoded"].to_numpy()
    classes = np.sort(np.unique(labels))
    class_to_position = {int(class_label): idx for idx, class_label in enumerate(classes)}

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    oof_predictions = np.full(len(df_label), fill_value=-1, dtype=int)
    oof_probabilities = np.zeros((len(df_label), len(classes)), dtype=float)
    fold_metrics: list[dict[str, float | int]] = []

    for fold, (train_idx, valid_idx) in enumerate(skf.split(uuids, labels), start=1):
        logger.info("%s: fold %s/%s", model_name, fold, n_folds)

        train_uuids = uuids[train_idx]
        valid_uuids = uuids[valid_idx]

        x_train = df_data.loc[
            df_data.index.get_level_values("uuid").isin(train_uuids)
        ]
        x_valid = df_data.loc[
            df_data.index.get_level_values("uuid").isin(valid_uuids)
        ]

        y_train = df_label.loc[df_label["uuid"].isin(train_uuids), "class_encoded"].to_numpy()
        y_valid = df_label.loc[df_label["uuid"].isin(valid_uuids), "class_encoded"].to_numpy()

        # build_sklearn_pipeline implements a list of feature engineering in a split aware manner
        feature_pipeline = build_sklearn_pipeline(feature_engineering_params)
        x_train_transformed = feature_pipeline.fit_transform(x_train)
        x_valid_transformed = feature_pipeline.transform(x_valid)

        # Fit one cv fold for the given classification model
        model = fit_one_fold(
            classifier_spec.build_model,
            x_train_transformed,
            y_train,
            x_valid_transformed,
            y_valid,
            {
                "model_params": model_params,
                "fit_params": fit_params,
            },
            use_eval_set=classifier_spec.supports_eval_set,
        )

        y_valid_pred = model.predict(x_valid_transformed)
        oof_predictions[valid_idx] = y_valid_pred

        if classifier_spec.supports_predict_proba:
            fold_probabilities = model.predict_proba(x_valid_transformed)
            for source_idx, class_label in enumerate(model.classes_):
                target_idx = class_to_position[int(class_label)]
                oof_probabilities[valid_idx, target_idx] = fold_probabilities[:, source_idx]

        fold_metrics.append(
            {
                "fold": fold,
                "validation_accuracy": float(accuracy_score(y_valid, y_valid_pred)),
                "validation_macro_f1": float(f1_score(y_valid, y_valid_pred, average="macro")),
                "validation_target_f1": float(
                    f1_score(y_valid, y_valid_pred, labels=[target_label], average="weighted")
                )
            }
        )

    if np.any(oof_predictions < 0):
        raise RuntimeError("Cross-validation did not generate predictions for all examples.")

    mean_target_f1 = float(
        np.mean([float(metric["validation_target_f1"]) for metric in fold_metrics])
    )

    return {
        "classes": classes.tolist(),
        "labels": labels.tolist(),
        "predictions": oof_predictions.tolist(),
        "probabilities": oof_probabilities.tolist(),
        "fold_metrics": fold_metrics,
        "selection_metric": mean_target_f1,
    }
