from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

@dataclass(frozen=True, slots=True)
class SegmentationConfig:
    input_bucket: str = "regenorganics-prioritydistributorbuffer-tiffs"
    output_bucket: str = "regenorganics-prioritydistributorbuffer-delineated"
    params_path: Optional[Path] = None
    model_ckpt_path: Optional[Path] = None
    input_file_ext: str = ".tiff"
    output_file_ext: str = ".csv"
    use_gpu: bool = True
