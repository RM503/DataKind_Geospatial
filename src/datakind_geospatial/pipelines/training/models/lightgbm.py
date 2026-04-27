from __future__ import annotations

from typing import Any 

from lightgbm import LGBMClassifier

def build_model(params: dict[str, Any]) -> LGBMClassifier:
    return LGBMClassifier(**(params or {}))

def supports_eval_set() -> bool:
    return True
