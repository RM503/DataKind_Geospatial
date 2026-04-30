from __future__ import annotations

import json
from pathlib import Path

import ee
import geopandas as gpd
import shapely
from shapely.geometry import Polygon


def return_ee_geometry(polygon: Polygon) -> ee.Geometry:
    """Converts shapely polygon to ee.Geometry object."""
    if not polygon.is_valid:
        polygon = shapely.make_valid(polygon)

    roi = ee.Geometry(json.loads(shapely.to_geojson(polygon)))
    return roi


def gdf_to_feature_collection(gdf: gpd.GeoDataFrame) -> ee.FeatureCollection:
    """Converts geopandas dataframe to ee.FeatureCollection."""
    features: list[ee.Feature] = []

    for _, row in gdf.iterrows():
        feature = ee.Feature(row["roi"])
        metadata = {
            "region": row["region"],
            "tile_name": row["tile_name"],
            "uuid": row["uuid"]
        }

        feature.set("metadata", metadata)
        features.append(feature)

    return ee.FeatureCollection(features)


def prepare_gdf(path: Path) -> gpd.GeoDataFrame:
    try:
        gdf = (
            gpd.read_file(path)
               .to_crs(epsg=4326)
               .assign(roi=lambda df: df.geometry.apply(return_ee_geometry))
        )

        return gdf
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found at path: {path}")
