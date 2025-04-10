# Running segment-geospatial on AWS EC2 instances

This directory contains modules for running `segment-geospatial` on AWS EC2 instances. Here we provide a description of the minimal setup required to run code on an EC2 instance.

## Requirements

For the code to run properly, one requires raster tiles of a particular geographic region in the form of `GeoTIFF` files. We suggest that these rasters be stored in an AWS S3 bucket from which they can be retrieve inside an EC2 instance. 

Speaking of EC2 instance, since `segment-geospatial` is resource intensive, the instance should be one with GPU capabilities, the simplest one of which is the `g4dn.xlarge` instance (or `g6.xlarge` for better performance). Furthermore, in order to keep the setup minimal, we recommend using a 'Deep Learning AMI' (DLAMI) which comes with CUDA preinstalled. 

## Package installations

The number of packages required are minimal in that `segment-geospatial` installs a large number of them as part of its dependencies. If the workflow includes only downloading and uploading to and from S3 buckets and running segmentation, running the `setup.sh` file will suffice. It contains the following code

```bash
# !/bin/bash

cd ~

curl -O https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh
bash ~/Anaconda3-2024.10-1-Linux-x86_64.sh

cd anaconda3/bin
echo 'export PATH=~/anaconda3/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

cd ~

conda create -n samgeo python
conda activate samgeo
conda install -c conda-forge mamba
mamba install -c conda-forge segment-geospatial "pytorch=*=cuda*"

conda install boto3
```
which downloads and installs the `conda` environment, adds its environment variable. Then, `segment-geospatial` is installed inside the conda environment (it was found that `pip` installation of the library does not work in the EC2 intance).

## Running the code

The `segmentation.py` code relies on the raster files being already present in an S3 bucket and another S3 bucket to which the segmentation results are upload. As such, the `input_bucket` and `output_bucket` information are passed as arguments when the script is executed

```
python3 segmentation.py --input_bucket <input_bucket> --output_bucket <output_bucket>
```
As the code is executed on each raster tile, the segmentation results are initially produced as GeoPackage files. The directory structure of the output are representative of that in the output bucket.
