from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import ee

IndexFunction = Callable[[ee.Image], ee.Image]


@dataclass(frozen=True, slots=True)
class VegetationIndexSpec:
    """Specifications for VI index retrieval"""
    name: str
    band_name: str
    add_band: IndexFunction
    scale: int = 10


def add_ndvi(img: ee.Image) -> ee.Image:
    ndvi = img.normalizedDifference(["B8", "B4"]).rename("ndvi")
    return img.addBands([ndvi])


def add_ndmi(img: ee.Image) -> ee.Image:
    ndmi = img.normalizedDifference(["B8", "B11"]).rename("ndmi")
    return img.addBands([ndmi])


def add_evi(img: ee.Image) -> ee.Image:
    evi = img.expression(
        "2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))",
        {
            "NIR": img.select("B8").divide(10_000),
            "RED": img.select("B4").divide(10_000),
            "BLUE": img.select("B2").divide(10_000)
        },
    ).rename("evi")
    return img.addBands([evi])


VEGETATION_INDEX_REGISTRY: dict[str, VegetationIndexSpec] = {
    "ndvi": VegetationIndexSpec(
        name="ndvi",
        band_name="ndvi",
        add_band=add_ndvi
    ),
    "ndmi": VegetationIndexSpec(
        name="ndmi",
        band_name="ndmi",
        add_band=add_ndmi
    ),
    "evi": VegetationIndexSpec(
        name="evi",
        band_name="evi",
        add_band=add_evi
    )
}


def get_vegetation_index(index_type: str) -> VegetationIndexSpec:
    key = index_type.lower()

    try:
        return VEGETATION_INDEX_REGISTRY[key]
    except KeyError as e:
        supported = ", ".join(VEGETATION_INDEX_REGISTRY.keys())
        raise ValueError(
            f"Unsupported vegetation index: {index_type}. "
            f"Supported indexes are: {supported}"
        ) from e
