from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

@dataclass(slots=True)
class BaseConfig:
    """
    Base configuration class for local AWS SageMaker jobs

    Attributes:
        region_name (str): AWS region name; defaults to 'us-east-1'
        profile_name (str): AWS profile name; defaults to 'default'
    """
    region_name: str = os.getenv("AWS_REGION", "us-east-1")
    profile_name: Optional[str] = os.getenv("AWS_PROFILE")

@dataclass(slots=True)
class PathConfig:
    """
    Base configuration class for temporary storage for GeoTIFF files
    during segmentation inside SageMaker environment.

    Attributes:
        scratch_dir (Path): Path to the temporary scratch directory
        log_dir (Path): Path to the temporary log directory
    """
    scratch_dir: Path = Path(os.getenv("SEGMENTATION_SCRATCH_ROOT", "/tmp/segmentation"))
    log_dir: Path = Path(os.getenv("SEGMENTATION_LOG_DIR", "/tmp/segmentation/logs"))

    def ensure_directories(self) -> None:
        self.scratch_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)