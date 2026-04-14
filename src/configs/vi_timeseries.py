"""
This module contains configuration class for VI time-series generation using
Google Earth Engine API.
"""
from __future__ import annotations

from datetime import datetime
from dataclasses import dataclass
from typing import Literal, Optional

from dateutil.relativedelta import relativedelta

VegetationIndexName = Literal["ndvi", "ndmi", "evi"]
CompositeReducer = Literal["median", "mean"]

@dataclass(frozen=True, slots=True)
class VISeriesConfig:
    """
    Configuration class for VI time-series generation

    Atrributes:
        gee_projcet (str, optional): the GEE project name
        collection_id (str): image collection name
        start_date (str): start date of data aquisition
        end_date (str): end date of data aquisition
        interval_days (int, optional): number of days between consecutive data points
        index_type (VegetationIndexName): name of the VI index to aquire
        cloud_prob_thresh (int): cloud probability threshold
        snow_prob_thresh (int): snow probability threshold
        composite_reducer (CompositeReducer): composite reducer (how to aggregate values across pixels)
        scale (int, optional): scaling factor
        geometry_crs (str): geometry CRS
    """
    # GEE/collection
    gee_project: Optional[str] = None
    collection_id: str = "COPERNICUS/S2_SR_HARMONIZED"

    # Temporal window
    start_date: str = (datetime.now() - relativedelta(years=5)).strftime("%Y-%m-%d")
    end_date: str = datetime.now().strftime("%Y-%m-%d")
    interval_days: Optional[int] = 10

    # Vegetation index
    index_type: VegetationIndexName = "ndvi"

    # Cloud/snow masking
    cloud_prob_thresh: int = 30
    snow_prob_thresh: int = 30

    # Reducer and export handing
    composite_reducer: CompositeReducer = "median"
    scale: Optional[int] = None
    geometry_crs: str = "EPSG:4326"

    def __post_init__(self) -> None:
        if self.interval_days < 0:
            raise ValueError("interval_days must be positive.")
        if not 0 <= self.cloud_prob_thresh <= 100:
            raise ValueError("cloud_prob_thresh must be between 0 and 100.")
        if not 0 <= self.snow_prob_thresh <= 100:
            raise ValueError("snow_prob_thresh must be between 0 and 100.")