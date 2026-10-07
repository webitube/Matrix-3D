#!/usr/bin/env bash
# Diagnostic: re-run simple-knn build and capture the real compiler error.
source /home/ron/miniconda3/etc/profile.d/conda.sh
conda activate matrix3d
export CUDA_HOME=/home/ron/cuda-12.6
export PATH=/home/ron/cuda-12.6/bin:$PATH
export LD_LIBRARY_PATH=/home/ron/cuda-12.6/lib64:$LD_LIBRARY_PATH
cd /mnt/g/Dev/Matrix-3D/submodules/simple-knn
python setup.py install > /tmp/simpleknn_build.log 2>&1
echo "exit code: $?"
echo "=== compiler errors ==="
grep -n -E "error:|fatal error|Error [0-9]|undefined reference" /tmp/simpleknn_build.log | head -20
echo "=== last 30 lines ==="
tail -30 /tmp/simpleknn_build.log
