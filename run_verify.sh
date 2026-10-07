#!/bin/bash
# Verify Matrix-3D environment in WSL
source /home/ron/miniconda3/etc/profile.d/conda.sh
conda activate matrix3d
export PATH=/home/ron/cuda-12.6/bin:$PATH
export LD_LIBRARY_PATH=/home/ron/cuda-12.6/lib64:$LD_LIBRARY_PATH
python /mnt/g/Dev/Matrix-3D/verify_env.py 2>&1 | tail -60
