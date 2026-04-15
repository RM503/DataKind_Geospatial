from __future__ import annotations

import os

import ee

from .vegetation_indices import VegetationIndexSpec, get_vegetation_index
from src.common.logging_config import get_logger
from src.common.ee_initialize import initialize_ee

logger = get_logger(__name__)

initialize_ee(os.getenv("GEE_PROJECT"))

def mask_cloud_and_shadow(
        img: ee.Image,
        cloud_prob_thresh: int = 30,
        snow_prob_thresh: int = 30
) -> ee.Image:
    """
    This function masks cloud and shadow of an image using
    GEE scene classification layer (SCL)
    """
    cloud_prob = img.select("MSK_CLDPRB")
    snow_prob = img.select("MSK_SNWPRB")
    cloud = cloud_prob.lt(cloud_prob_thresh)
    snow = snow_prob.lt(snow_prob_thresh)

    # Use SCL to select shadows and cirrus cloud masks
    scl = img.select("SCL")
    shadow_mask = scl.eq(3)
    cirrus_mask = scl.eq(10)

    mask = cloud.And(snow.And(cirrus_mask.neq(1))).And(shadow_mask.neq(1))

    return img.updateMask(mask)