"""
Model registry for all the classifiers used in experimentations and
training. This relies on all classifiers having a common SKLearn API
that involves initialization and fit.
"""
from __future__ import annotations 

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .lightgbm import build_model as build_lgbm
from .xgboost import build_model as build_xgboost

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class ClassifierSpec:
    name: str 
    build_model: Callable[[dict[str, Any]], Any]
    supports_predict_proba: bool = True 
    supports_eval_set: bool = False

# Build classifier catalog
CLASSIFIERS = {
    "lightgbm": ClassifierSpec(
        name="lightgbm",
        build_model=build_lgbm,
        supports_predict_proba=True,
        supports_eval_set=True
    ),
    "xgboost": ClassifierSpec(
        name="xgboost",
        build_model=build_xgboost,
        supports_predict_proba=True,
        supports_eval_set=True
    )
}

def get_classifier(name: str) -> ClassifierSpec:
    try:
        return CLASSIFIERS[name]
    except KeyError as exc:
        supported = ", ".join(sorted(CLASSIFIERS))
        raise KeyError(f"Unknown classifier '{name}'. Supported classifiers: {supported}") from exc
