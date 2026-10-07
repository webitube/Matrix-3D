#!/usr/bin/env bash
# Rebuild the two failed CUDA extensions with full error output.
source /home/ron/miniconda3/etc/profile.d/conda.sh
conda activate matrix3d
export CUDA_HOME=/home/ron/cuda-12.6
export PATH=/home/ron/cuda-12.6/bin:$PATH
export LD_LIBRARY_PATH=/home/ron/cuda-12.6/lib64:$LD_LIBRARY_PATH
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
unset CUDA_CFLAGS CUDA_LDFLAGS

echo "=== [A] diff-gaussian-rasterization-w-pose (full error) ==="
cd /mnt/g/Dev/Matrix-3D
pip install --no-build-isolation git+https://github.com/rmurai0610/diff-gaussian-rasterization-w-pose.git > /tmp/dgr_build.log 2>&1
echo "exit: $?"
grep -n -E "error:|fatal error|Error [0-9]|undefined reference|No such file" /tmp/dgr_build.log | head -15
echo "--- last 15 lines ---"
tail -15 /tmp/dgr_build.log

echo
echo "=== [B] ODGS odgs_gaussian_rasterization (full error) ==="
cd /mnt/g/Dev/Matrix-3D/submodules/ODGS
pip install --no-build-isolation submodules/odgs-gaussian-rasterization > /tmp/odgs_build.log 2>&1
echo "exit: $?"
grep -n -E "error:|fatal error|Error [0-9]|undefined reference|No such file" /tmp/odgs_build.log | head -15
echo "--- last 15 lines ---"
tail -15 /tmp/odgs_build.log
