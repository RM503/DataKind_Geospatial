import logging
from typing import Self

import numpy as np 
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)

class RemoveNanColumns(TransformerMixin, BaseEstimator):
    """
    Implements a transformer inside an SKLearn pipeline which
    removes columns where all elements are NaN. This may happen
    when time-series data are converted to features via Catch-22.
    """
    def __init__(self) -> None:
        self.columns_to_drop: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray=None) -> Self:
        x = self._validate_input(x)
        nan_cols = np.where(np.all(np.isnan(x), axis=0))[0]

        if len(nan_cols) > 0:
            logger.info("Removing columns with all NaN values.")
        self.columns_to_drop = nan_cols 
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = self._validate_input(x)
        if self.columns_to_drop is None or len(self.columns_to_drop) == 0:
            return x.copy()
        return np.delete(x, self.columns_to_drop, axis=1)

    def _validate_input(self, x: np.ndarray) -> np.ndarray:
        if isinstance(x, np.ndarray):
            return x 
        else:
            raise TypeError(
                f"Object '{x}' does not have the correct type;"
                f"expected type np.ndarray, got {type(x)} instead"
            )
