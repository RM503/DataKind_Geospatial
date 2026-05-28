"""
Main module for generating VI time-series data from Earth Engine, along with
formatting final dataset shape.
"""

from __future__ import annotations

import os
from datetime import datetime

import ee

from .vegetation_indices import VegetationIndexSpec
from src.common.logging_config import get_logger
from src.common.ee_initialize import initialize_ee

logger = get_logger(__name__)

initialize_ee(os.getenv("GEE_PROJECT"))

IMAGE_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"

def mask_cloud_and_shadow(
        image: ee.Image,
        cloud_prob_thresh: int = 30,
        snow_prob_thresh: int = 30
) -> ee.Image:
    """
    Performs cloud and shadow masking on a single Earth Engine image.

    Args:
        (i) image (ee.Image): Earth Engine image on which masking should be performed
        (ii) cloud_prob_thresh (int): Cloud probability threshold; defaults to 30
        (iii) snow_prob_thresh (int): Snow probability threshold; defaults to 30

    Returns:
        (ee.Image): Earth Engine image with update masks.
    """
    cloud_prob = image.select("MSK_CLDPRB")
    snow_prob = image.select("MSK_SNWPRB")
    cloud = cloud_prob.lt(cloud_prob_thresh)
    snow = snow_prob.lt(snow_prob_thresh)

    # Use SCL to select shadows and cirrus cloud masks
    scl = image.select("SCL")
    shadow_mask = scl.eq(3)
    cirrus_mask = scl.eq(10)

    mask = cloud.And(snow.And(cirrus_mask.neq(1))).And(shadow_mask.neq(1))

    return image.updateMask(mask)


def make_index_collection(
        start_date: str,
        end_date: str,
        feature_collection: ee.FeatureCollection,
        index_spec: VegetationIndexSpec
) -> ee.ImageCollection:
    """
    Builds a feature collection of vegetation-index means for each image and region.

    Filters Sentinel-2 surface reflectance imagery to the requested date range and
    feature bounds, applies cloud/shadow masking, adds the configured vegetation
    index band, and computes the spatial mean for each feature in
    `feature_collection`. The returned collection is flattened so each feature
    represents one region-image observation with a `date` property.

    Args:
        (i) start_date (str): Start date for the image query in `YYYY-MM-DD` format.
        (ii) end_date (str): End date for the image query in `YYYY-MM-DD` format.
        (iii) feature_collection (ee.FeatureCollection): Regions over which index means
            are reduced.
        (iv) index_spec (VegetationIndexSpec): Vegetation index configuration, including
            band name, band-generation function, and reduction scale.

    Returns:
        *ee.ImageCollection): Flattened Earth Engine collection of per-region,
        per-date vegetation-index mean features.
    """
    if datetime.strptime(start_date, "%Y-%m-%d") < datetime(2016, 6, 13):
        raise ValueError("Start date must be before end date.")
    if not isinstance(feature_collection, ee.FeatureCollection):
        raise TypeError("FeatureCollection must be of type ee.FeatureCollection")

    index_band_name = index_spec.band_name
    add_band_fn = index_spec.add_band

    image_collection = (
        ee.ImageCollection(IMAGE_COLLECTION).filterDate(start_date, end_date)
          .filterBounds(feature_collection).map(mask_cloud_and_shadow).map(add_band_fn)
    ).select(index_band_name)

    def map_index(image: ee.Image):
        """
        Applies a reducer to each image in the collection and performs a spatial mean of the index.
        """
        stats= image.reduceRegions(
            collection=feature_collection,
            reducer=ee.Reducer.mean().setOutputs([index_band_name]),
            scale=index_spec.scale
        ).filter(ee.Filter.neq(index_band_name, None)) # filters out null values

        def set_date(feature: ee.Feature) -> ee.Feature:
            return feature.set("date", image.date().format("YYYY-MM-dd"))
        return stats.map(set_date)

    return image_collection.map(map_index).flatten()


def format_table(
        table: ee.ImageCollection,
        row_id: str,
        col_id: str,
        index_name: str
) -> ee.FeatureCollection:
    """
    Pivots long vegetation-index observations into a wide feature collection.

    Groups features by `row_id` and converts matching `col_id` values into
    output properties containing the corresponding vegetation-index value. Missing
    index values are filled with `-9999`, and numeric values are formatted to
    three decimal places.

    Args:
        (i) table (ee.ImageCollection): Long-form collection containing one feature per
            row/column vegetation-index observation.
        (ii) row_id (str): Property used to identify each output row.
        (iii) col_id (str): Property whose values become output column names.
        (iv) index_name (str): Property containing the vegetation-index value to pivot.

    Returns:
        (ee.FeatureCollection): Wide-form collection with one feature per `row_id`
        and one property per observed `col_id`.
    """
    rows = table.distinct(row_id)
    condition = ee.Filter.equals(leftField=row_id, rightField=row_id)
    joined = ee.Join.saveAll("matches").apply(
        primary=rows,
        secondary=table,
        condition=condition
    )

    def map_row(row: ee.Feature) -> ee.Feature:
        matches = ee.List(row.get("matches"))

        def extract_values(feature: ee.Feature) -> ee.Feature:
            feature = ee.Feature(feature)
            vi_value = ee.List([feature.get(index_name), -9999]).reduce(ee.Reducer.firstNonNull())
            return [feature.get(col_id), ee.Number(vi_value).format("%.3f")]

        values = matches.map(extract_values)
        flat_values = ee.Dictionary(ee.List(values).flatten())
        return row.select([row_id]).set(flat_values)

    return ee.FeatureCollection(joined.map(map_row))
