#!/usr/bin/env bash
# Test: build simple-knn with explicit system gcc (bypass conda gcc 15.2 shadowing).
source /home/ron/miniconda3/etc/profile.d/conda.sh
conda activate matrix3d
export CUDA_HOME=/home/ron/cuda-12.6
export PATH=/home/ron/cuda-12.6/bin:$PATH
export LD_LIBRARY_PATH=/home/ron/cuda-12.6/lib64:$LD_LIBRARY_PATH
# Force system gcc 13.3 (CUDA 12.6 supports gcc <= 13; conda env has gcc 15.2)
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
cd /mnt/g/Dev/Matrix-3D/submodules/simple-knn
rm -rf build
python setup.py install > /tmp/simpleknn_test.log 2>&1
echo "exit code: $?"
echo "=== errors (if any) ==="
grep -n -E "error:|fatal error|Error [0-9]" /tmp/simpleknn_test.log | head -10
echo "=== import test ==="
python -c "import simple_knn; print('simple_knn OK')" 2>&1 | tail -3
