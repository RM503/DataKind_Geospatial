#!/bin/bash

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