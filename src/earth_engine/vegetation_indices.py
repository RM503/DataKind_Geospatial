"""
Module for registering VI indices.
"""

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


# Define functional forms of registered indices
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


def add_ndre(image: ee.Image) -> ee.Image:
    ndre = image.normalizedDifference(["B8A", "B5"]).rename("NDRE")
    return image.addBands(ndre)


def add_savi(image: ee.Image, L: float = 0.5) -> ee.Image:
    nir  = image.select("B8")
    red  = image.select("B4")

    savi = (
        nir.subtract(red)
           .divide(nir.add(red).add(L))
           .multiply(1 + L)
           .rename("SAVI")
    )
    return image.addBands(savi)


def add_bsi(image: ee.Image) -> ee.Image:
    swir1 = image.select("B11")
    red   = image.select("B4")
    nir   = image.select("B8")
    blue  = image.select("B2")

    numerator   = swir1.add(red).subtract(nir.add(blue))
    denominator = swir1.add(red).add(nir.add(blue))

    bsi = numerator.divide(denominator).rename("BSI")
    return image.addBands(bsi)


# Register vegetation indices
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
    ),
    "ndre": VegetationIndexSpec(
        name="ndre",
        band_name="ndre",
        add_band=add_ndmi
    ),
    "savi": VegetationIndexSpec(
        name="savi",
        band_name="savi",
        add_band=add_savi
    ),
    "bsi": VegetationIndexSpec(
        name="bsi",
        band_name="bsi",
        add_band=add_bsi
    )
}


def get_vegetation_index(index_name: str) -> VegetationIndexSpec:
    """Returns `VegetationIndexSpec` for given index_name."""
    key = index_name.lower()

    try:
        return VEGETATION_INDEX_REGISTRY[key]
    except KeyError as e:
        supported = ", ".join(VEGETATION_INDEX_REGISTRY.keys())
        raise ValueError(
            f"Unsupported vegetation index: {index_name}. "
            f"Supported indexes are: {supported}"
        ) from e
