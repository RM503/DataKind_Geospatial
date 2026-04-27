from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

def fit_one_fold(
    model_builder,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    params: dict[str, Any],
    *,
    use_eval_set: bool = False,
):
    model_params = params.get("model_params", {})
    fit_params = params.get("fit_params", {}).copy()

    model = model_builder(model_params)

    if use_eval_set:
        fit_params.setdefault("eval_set", [(x_valid, y_valid)])

    model.fit(
        x_train,
        np.asarray(y_train),
        **fit_params,
    )

    return model
