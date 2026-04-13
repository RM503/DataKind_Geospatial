from __future__ import annotations 

import argparse 

import boto3 
from sagemaker.core.helper.session_helper import Session
from sagemaker.core.processing import ScriptProcessor

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Submit a SageMaker processing job for SamGeo segmentation.")

    parser.add_argument("--role-arn", required=True)
    parser.add_argument("--image-uri", required=True)
    parser.add_argument("--instance-type", default="ml.g5.2xlarge")
    parser.add_argument("--instance-count", type=int, default=1)

    parser.add_argument("--input-bucket", required=True)
    parser.add_argument("--output-bucket", required=True)
    parser.add_argument("--output-prefix", default="segmentation")

    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--params-path", default=None)
    parser.add_argument("--regions", default=None)
    parser.add_argument("--input-suffix", default=".tiff")

    parser.add_argument("--volume-size-gb", type=int, default=50)
    parser.add_argument("--max-runtime-seconds", type=int, default=86400)

    parser.add_argument("--job-name", default=None)
    parser.add_argument("--region-name", default=None)

    parser.add_argument("--emit-mask-tiff", action="store_true")
    parser.add_argument("--emit-gpkg", action="store_true")
    parser.add_argument("--emit-csv", action="store_true")

    return parser

def main() -> None:
    args = build_parser().parse_args()
    print("✓ Args parsed", flush=True)

    boto_session = boto3.Session(region_name=args.region_name)
    print("✓ Boto session created", flush=True)

    sagemaker_session = Session(boto_session=boto_session)
    print("✓ SageMaker session created", flush=True)

    processor = ScriptProcessor(
        role=args.role_arn,
        image_uri=args.image_uri,
        command=["python"], # run as Python script
        instance_type=args.instance_type,
        instance_count=args.instance_count,
        volume_size_in_gb=args.volume_size_gb,
        max_runtime_in_seconds=args.max_runtime_seconds,
        sagemaker_session=sagemaker_session
    )
    print("✓ Processor created", flush=True)

    # These are commands that run inside the container
    # Runs src.segmentation.cli and appends arguments
    arguments = [
        #"-m", "src.segmentation.cli",
        "--input-bucket", args.input_bucket,
        "--output-bucket", args.output_bucket,
        "--output-prefix", args.output_prefix,
        "--checkpoint-path", args.checkpoint_path,
        "--input-suffix", args.input_suffix
    ]

    if args.params_path:
        arguments.extend(["--params-path", args.params_path])

    if args.regions:
        arguments.extend(["--regions", args.regions])

    if args.emit_mask_tiff:
        arguments.append("--emit-mask-tiff")

    if args.emit_gpkg:
        arguments.append("--emit-gpkg")

    if args.emit_csv:
        arguments.append("--emit-csv")

    print(f"✓ Submitting job with arguments: {arguments}", flush=True)

    processor.run(
        code="jobs/segmentation/processing_entry.py",
        job_name=args.job_name,
        arguments=arguments,
        wait=True,
        logs=True
    )

if __name__ == "__main__":
    main()