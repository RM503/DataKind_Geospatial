"""
Configurations for segmentation jobs
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from configs.base import AWSConfig, PathConfig

@dataclass(slots=True)
class ModelConfig:
    """Segmentation model specific configs"""
    checkpoint_path: Optional[Path] = None
    model_type: str = "vit_h"
    device: Optional[str] = None

@dataclass(slots=True)
class SAMConfig:
    """Segmentation parameter specific configs"""
    params_path: Optional[Path] = None
    batch: bool = False
    foreground: bool = True
    erosion_kernel: tuple[int, int] = (3, 3)
    mask_multiplier: int = 255

@dataclass(slots=True)
class IOConfig:
    """Segmentation I/O specific configs"""
    input_suffix: str = ".tiff"
    output_prefix: str = "segmentation"
    emit_mask_tiff: bool = True
    emit_gpkg: bool = True
    emit_csv: bool = True

@dataclass
class RuntimeConfig:
    cleanup_local_files: bool = True

@dataclass(slots=True)
class SegmentationConfig:
    input_bucket: str = "regenorganics-prioritydistributorbuffer-tiffs"
    output_bucket: str = "regenorganics-prioritydistributorbuffer-delineated"

    aws: AWSConfig = field(default_factory=AWSConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    model: Optional[ModelConfig] = None
    sam: SAMConfig = field(default_factory=SAMConfig)
    io: IOConfig = field(default_factory=IOConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def validate(self) -> None:
        if self.model is None:
            raise ValueError("ModelConfig is required.")
        if self.model.checkpoint_path is None:
            raise ValueError("checkpoint_path is required.")
