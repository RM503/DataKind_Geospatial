"""IO utilities for Earth Engine processing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.common.logging_config import get_logger

logger = get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ASSETS_CONFIG_PATH = REPO_ROOT / "configs" / "earth_engine" / "assets.yml"


def load_assets_config(config_path: Path | str = DEFAULT_ASSETS_CONFIG_PATH) -> dict[str, Any]:
    """Load the Earth Engine assets config outside Kedro's conf structure."""
    path = Path(config_path)

    with path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}

    if not isinstance(config, dict):
        raise ValueError(f"Expected a mapping in assets config: {path}")

    return config


def resolve_geometry_paths(config_path: Path | str = DEFAULT_ASSETS_CONFIG_PATH) -> list[Path]:
    """Return configured GPKG paths for either all assets or one selected asset."""
    config = load_assets_config(config_path)
    geometry_assets = config.get("geometry_assets", {})
    selection = geometry_assets.get("selection", {})
    gpkg_files = geometry_assets.get("gpkg_files", {})

    if not isinstance(selection, dict):
        raise ValueError("Expected geometry_assets.selection to be a mapping.")
    if not isinstance(gpkg_files, dict) or not gpkg_files:
        raise ValueError("Expected geometry_assets.gpkg_files to contain at least one GPKG path.")

    process_all = bool(selection.get("all", False))
    single_asset = selection.get("single")

    if process_all:
        return [Path(path) for path in gpkg_files.values()]

    if not single_asset:
        raise ValueError("Set geometry_assets.selection.all=true or provide selection.single.")

    if single_asset not in gpkg_files:
        available_assets = ", ".join(sorted(gpkg_files))
        raise ValueError(
            f"Unknown geometry asset '{single_asset}'. Available assets: {available_assets}"
        )

    return [Path(gpkg_files[single_asset])]
