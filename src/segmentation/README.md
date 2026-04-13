# **GeoTIFF segmentation pipeline**

The segmentation pipeline (`src/segmentation/`) contains scripts for performing naive field delineation on raster tiles. It leverages AWS SageMaker's `ScriptProcessor` module for performing segmentation jobs from a local machine, ingesting GeoTIFF files from an `input-bucket` and writing segmentation artifacts to an `output-bucket`.

## **Directory Structure**

The directory structure for segmentations jobs is defined as follows —
```
datakind_geospatial/
|—— congigs/
|   └── segmentation/
|      └── params.json
|—— docker/
|   └── segmentation/
|      └── Dockerfile
|      └── requirements.txt
|—— jobs/
|   └── segmentation/
|      └── processing_entry.py
|      └── run_processing.py
|—— src/
|   └── configs/
|      └── segmentation.py
|   └── segmentation/
|      └── cli.py
|      └── pipeline.py
|      └── s3_io.py
|      └── sam_model.py
```

### **Main Scripts**

The `src/segmentation` directory contains all the processing scripts required by the segmentation algorithm — including model initialization, S3 I/O functionalities, pipeline patching together all operations and a cli for passing command line arguments. The `src/configs` directory contains the segmentation configuration class, which is utilized by the pipeline for properly storing parameters and configurations.

### **Jobs and Script Entry Point**

The main entry-point of the processing jobs is located in `jobs/segmentation/run_processing_job.py`, where the `ScriptProcessor` instance executes `jobs/segmentation/processing_entry.py` — the latter, in turn, runs `src/segmentation/cli` from which the segmentation initialization and processing scripts are called.

To be able to start a processing job, a docker image needs to be pushed to AWS ECR. At a minimum, the docker image should contain the `configs/segmentation`, `docker/segmentation`, `jobs/segmentation`, `src/common`, `src/configs` and `src/segmentation` directories, along with necessary packages for performing geospatial data processing. The processing job can then be submitted as follows —

```
uv run jobs/segmentation/run_processing_job.py \
--role-arn arn:aws:iam:xxxxxxxxxxxx/service-role/AmazonSageMaker-ExecutionRole-xxxxxxxxxxxxxxx \
--instance-type ml.g5.xlarge \
--input-bucket input-bucket-name \
--output-bucket output-bucket-name \
--output-prefix segmentation \
--regions region1,region2,... \
--checkpoint-path /opt/program/checkpoints/sam_vit_h_4b8939.pth \
--params-path /opt/program/configs/segmentation/params.json \
--emit-csv \
--emit-gpkg
```