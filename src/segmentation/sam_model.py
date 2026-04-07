from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch
from samgeo import SamGeo

from common.logging_config import get_logger

logger = get_logger(__name__)

def resolve_device(explicit_device: Optional[str]=None) -> str:
    if explicit_device:
        return explicit_device
    return "cuda" if torch.cuda.is_available() else "cpu"

def build_samgeo(
        checkpoint_path: Path,
        sam_kwargs: dict[str, Any],
        model_type: str = "vit_h",
        device: Optional[str] = None
) -> SamGeo:
    resolved_device = resolve_device(device)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"SAM checkpoint not found at: {checkpoint_path}")

    logger.info(f"Initializing SamGeo with device={device}, checkpoint={checkpoint_path}")

    sam = SamGeo(
        model_type=model_type,
        checkpoint=str(checkpoint_path),
        device=resolved_device,
        sam_kwargs=sam_kwargs
    )

    return sam

def cleanup_torch_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()