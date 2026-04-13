# Dockerfile for submitting AWS SageMaker ScriptProcessor jobs
# Can be obtained from AWS ECR

FROM --platform=linux/amd64 pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime

ENV PYTHONBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/opt/program
ENV PIP_BREAK_SYSTEM_PACKAGES=1

WORKDIR /opt/program 

# Install GDAL for geospatial capabilities
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Download SAM checkpoint from HuggingFace
RUN python -m pip install --no-cache-dir huggingface_hub && \
    python -c "from huggingface_hub import hf_hub_download; \
              hf_hub_download('HCMUE-Research/SAM-vit-h', 'sam_vit_h_4b8939.pth', local_dir='/opt/program/checkpoints')"

COPY docker/segmentation/requirements.txt /opt/program/requirements.txt
RUN python -m pip install --no-cache-dir -r /opt/program/requirements.txt

# Require only src/segmentation and src/common (for logging utils)
COPY /configs/segmentation /opt/program/configs/segmentation
COPY jobs /opt/program/jobs
COPY /src/common /opt/program/src/common
COPY /src/configs /opt/program/src/configs
COPY /src/segmentation /opt/program/src/segmentation

ENTRYPOINT ["python", "jobs/segmentation/processing_entry.py"]