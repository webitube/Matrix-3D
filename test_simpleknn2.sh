#!/usr/bin/env bash
# Test: build simple-knn with system gcc + forced cfloat include (no source edit).
source /home/ron/miniconda3/etc/profile.d/conda.sh
conda activate matrix3d
export CUDA_HOME=/home/ron/cuda-12.6
export PATH=/home/ron/cuda-12.6/bin:$PATH
export LD_LIBRARY_PATH=/home/ron/cuda-12.6/lib64:$LD_LIBRARY_PATH
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export NVCC_FLAGS="-include cfloat"
cd /mnt/g/Dev/Matrix-3D/submodules/simple-knn
rm -rf build
python setup.py install > /tmp/simpleknn_test2.log 2>&1
echo "exit code: $?"
echo "=== errors (if any) ==="
grep -n -E "error:|fatal error|Error [0-9]" /tmp/simpleknn_test2.log | head -10
echo "=== real import test (simple_knn._C) ==="
python -c "from simple_knn._C import distCUDA2; print('simple_knn._C OK')" 2>&1 | tail -3
