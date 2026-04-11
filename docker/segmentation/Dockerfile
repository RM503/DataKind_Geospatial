FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

WORKDIR /opt/program 

RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY docker/segmentation/requirements.txt /opt/program/requirements.txt
RUN pip install --no-cache-dir -r /opt/program/requirements.txt

COPY /src/segmentation /opt/program/src/segmentation 
COPY jobs /opt/program/jobs

ENV PYTHONPATH=/opt/program

ENTRYPOINT ["python", "jobs/segmentation/processing_entry.py"]