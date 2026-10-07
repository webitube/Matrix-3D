#!/usr/bin/env bash
source /home/ron/miniconda3/etc/profile.d/conda.sh
conda activate matrix3d
export CUDA_HOME=/home/ron/cuda-12.6
export PATH=/home/ron/cuda-12.6/bin:$PATH
export LD_LIBRARY_PATH=/home/ron/cuda-12.6/lib64:$LD_LIBRARY_PATH
python -c "import torch; from simple_knn._C import distCUDA2; print('simple_knn._C OK')" 2>&1 | tail -3
