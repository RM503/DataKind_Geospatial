"""
This is a CLI tool for passing parameters and configurations to build
SamGeo processing job.
"""

from __future__ import annotations 

import argparse 

from src.common.logging_config import get_logger
from src.configs.segmentation import SegmentationConfig
from src.segmentation.pipeline import run_all_regions

logger = get_logger(__name__)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SamGeo segmentation over GeoTIFFS.")

    parser.add_argument("--input-bucket", required=True)
    parser.add_argument("--output-bucket", required=True)

    parser.add_argument("--output-prefix", default="segmentation")
    parser.add_argument("--input-suffix", default=".tiff")
    parser.add_argument("--regions", default=None, help="Comma-separated list of regions to process.")

    parser.add_argument("--params-path", default=None)
    parser.add_argument("--checkpoint-path", required=True)

    parser.add_argument("--batch", action="store_true")

    parser.add_argument("--emit-mask-tiff", action="store_true")
    parser.add_argument("--emit-gpkg", action="store_true")
    parser.add_argument("--emit-csv", action="store_true")

    parser.add_argument("--cleanup-local-files", action="store_true", default=True)

    return parser 

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not any([args.emit_mask_tiff, args.emit_gpkg, args.emit_csv]):
        args.emit_csv = True

    config = SegmentationConfig.from_args(args)
    run_all_regions(config=config)

if __name__ == "__main__":
    main()