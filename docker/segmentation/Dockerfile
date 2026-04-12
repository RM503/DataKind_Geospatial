# Dockerfile for submitting AWS SageMaker ScriptProcessor jobs
# Can be obtained from AWS ECR

FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

WORKDIR /opt/program 

# Install GDAL for geospatial capabilities
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Download SAM checkpoint from HuggingFace
RUN pip install huggingface_hub && \
    python -c "from huggingface_hub import hf_hub_download; \
              hf_hub_download('HCMUE-Research/SAM-vit-h', 'sam_vit_h_4b8939.pth', local_dir='/opt/program/checkpoints')"

COPY docker/segmentation/requirements.txt /opt/program/requirements.txt
RUN pip install --no-cache-dir -r /opt/program/requirements.txt

# Require only src/segmentation and src/common (for logging utils)
COPY /configs/segmentation /opt/program/configs/segmentation
COPY jobs /opt/program/jobs
COPY /src/common /opt/program/src/common
COPY /src/configs /opt/program/src/configs
COPY /src/segmentation /opt/program/src/segmentation

ENV PYTHONPATH=/opt/program

ENTRYPOINT ["python", "jobs/segmentation/processing_entry.py"]