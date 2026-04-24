"""Module for VI time-series dataframe validation."""
from __future__ import annotations 

import pandas as pd
import pandera as pa
from pandera import DataFrameSchema, Column 

class VIDataValidator:
    def __init__(self, vi_column: str) -> None:
        self.vi_column = vi_column
        self.schema = self._build_schema()

    def _build_schema(self) -> DataFrameSchema:
        """Returns the valid form of dataframe schema for VI time-series data."""
        return DataFrameSchema({
            "uuid": Column(str, nullable=False),
            "date": Column(pa.DateTime, nullable=False),
            self.vi_column: Column(float, checks=pa.Check.in_range(-1, 1), nullable=True)
        })

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.schema.validate(df)