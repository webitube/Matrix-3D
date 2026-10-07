#!/usr/bin/env bash
source /home/ron/miniconda3/etc/profile.d/conda.sh
conda activate matrix3d
export CUDA_HOME=/home/ron/cuda-12.6
export PATH=/home/ron/cuda-12.6/bin:$PATH
export LD_LIBRARY_PATH=/home/ron/cuda-12.6/lib64:$LD_LIBRARY_PATH
echo "=== FAIL lines ==="
python /mnt/g/Dev/Matrix-3D/verify_env.py 2>&1 | grep "^FAIL"
echo
echo "=== key extensions ==="
python /mnt/g/Dev/Matrix-3D/verify_env.py 2>&1 | grep -E "diff_gaussian|odgs|simple_knn|pytorch3d|taming|clip|basicsr|realesrgan|streamlit|diffsynth"
