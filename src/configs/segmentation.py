"""
This module contains configurations for segmentation jobs. The main `SegmentationConfig`
contains attributes defined by smaller dataclasses. Each class implements a `validate`
method for validating arguments passed to the configurations through the CLI.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .base import AWSConfig, PathConfig

@dataclass(slots=True)
class ModelConfig:
    """Segmentation model specific configs"""
    checkpoint_path: Optional[Path] = None
    model_type: str = "vit_h"
    device: Optional[str] = None

    def validate(self) -> None:
        if self.checkpoint_path is None:
            raise ValueError("Model checkpoint path is required.")

@dataclass(slots=True)
class SAMConfig:
    """Segmentation parameter specific configs"""
    params_path: Optional[Path] = None
    batch: bool = False
    foreground: bool = True
    erosion_kernel: tuple[int, int] = (3, 3)
    mask_multiplier: int = 255

    def load_sam_kwargs(self) -> dict[str, Any]:
        if self.params_path is None:
            return {}

        if not self.params_path.exists():
            raise FileNotFoundError(f"SAM params file not found: {self.params_path}")

        with self.params_path.open(mode="r", encoding="utf-8") as f:
            payload = json.loads(f)

        if "params" in payload and isinstance(payload["params"], list) and payload["params"]:
            return payload["params"][0]

        if isinstance(payload, dict):
            return payload

        raise ValueError("Unsupported params.json format.")

    def validate(self) -> None:
        if (
            not isinstance(self.erosion_kernel, tuple)
            or len(self.erosion_kernel) != 2
            or not all(isinstance(x, int) for x in self.erosion_kernel)
        ):
            raise ValueError("erosion_kernel must be a tuple of two integers.")

        if self.mask_multiplier <= 0:
            raise ValueError("mask_multiplier must be a positive integer.")

@dataclass(slots=True)
class IOConfig:
    """Segmentation I/O specific configs"""
    input_suffix: str = ".tiff"
    output_prefix: str = "segmentation"
    emit_mask_tiff: bool = True
    emit_gpkg: bool = True
    emit_csv: bool = True

    def validate(self) -> None:
        if not self.input_suffix.startswith("."):
            raise ValueError("input_suffix must start with '.'")
        if not self.output_prefix:
            raise ValueError("output_prefix must be a non-empty string.")
        if not any([self.emit_mask_tiff, self.emit_gpkg, self.emit_csv]):
            raise ValueError(
                "At least one output type must be enabled: "
                "emit_mask_tiff, emit_gpkg or emit_csv."
            )

@dataclass(slots=True)
class RuntimeConfig:
    cleanup_local_files: bool = True

    def validate(self) -> None:
        return

@dataclass(slots=True)
class SegmentationConfig:
    input_bucket: str = "regenorganics-prioritydistributorbuffer-tiffs"
    output_bucket: str = "regenorganics-prioritydistributorbuffer-delineated"
    region_allowlist: Optional[list[str]] = None

    aws: AWSConfig = field(default_factory=AWSConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    model: Optional[ModelConfig] = None
    sam: SAMConfig = field(default_factory=SAMConfig)
    io: IOConfig = field(default_factory=IOConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def validate(self) -> None:
        if not self.input_bucket:
            raise ValueError("input_bucket must be a non-empty string.")
        if not self.output_bucket:
            raise ValueError("output_bucket must be a non-empty string.")
        if not self.model:
            raise ValueError("model configuration is required")

        # Validation checks
        self.model.validate()
        self.sam.validate()
        self.io.validate()
        self.runtime.validate()

        self.paths.ensure_directories()

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "SegmentationConfig":
        """Builds SegmentationConfig from CLI arguments"""
        region_allowlist: Optional[list[str]] = None

        if getattr(args, "regions", None):
            region_allowlist = [item.strip() for item in args.regions.split(",") if item.strip()]

        model = ModelConfig(
            checkpoint_path=Path(args.checkpoint_path),
            model_type=getattr(args, "model_type", "vit_h"),
            device=getattr(args, "device", None)
        )

        sam = SAMConfig(
            params_path=Path(args.params_path) if getattr(args, "params_path", None) else None,
            batch=getattr(args, "batch", False),
            foreground=getattr(args, "foreground", True),
            erosion_kernel=tuple(getattr(args, "erosion_kernel", (3, 3))),
            mask_multiplier=getattr(args, "mask_multiplier", 255)
        )

        io = IOConfig(
            input_suffix=getattr(args, "input_suffix", ".tiff"),
            output_prefix=getattr(args, "output_prefix", "segmentation"),
            emit_mask_tiff=getattr(args, "emit_mask_tiff", True),
            emit_gpkg=getattr(args, "emit_gpkg", True),
            emit_csv=getattr(args, "emit_csv", True),
        )

        runtime = RuntimeConfig(
            cleanup_local_files=getattr(args, "cleanup_local_files", True),
        )

        config = cls(
            input_bucket=args.input_bucket,
            output_bucket=args.output_bucket,
            region_allowlist=region_allowlist,
            model=model,
            sam=sam,
            io=io,
            runtime=runtime,
        )
        config.validate()
        return config