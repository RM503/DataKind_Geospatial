from __future__ import annotations 

from typing import Any

from xgboost import XGBClassifier

def build_model(params: dict[str, Any]) -> XGBClassifier:
    return XGBClassifier(**(params or {}))

def supports_eval_set() -> bool:
    return True
