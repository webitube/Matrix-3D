#!/usr/bin/env bash
# Initialize the glm submodule, then build diff-gaussian-rasterization from a /tmp copy.
source /home/ron/miniconda3/etc/profile.d/conda.sh
conda activate matrix3d
export CUDA_HOME=/home/ron/cuda-12.6
export PATH=/home/ron/cuda-12.6/bin:$PATH
export LD_LIBRARY_PATH=/home/ron/cuda-12.6/lib64:$LD_LIBRARY_PATH
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
unset CUDA_CFLAGS CUDA_LDFLAGS

echo "=== Initialize glm submodule ==="
cd /mnt/g/Dev/Matrix-3D/submodules/diff-gaussian-rasterization-w-pose
git submodule update --init --recursive 2>&1 | tail -3
echo "glm.hpp present: $(ls third_party/glm/glm/glm.hpp 2>/dev/null || echo NO)"

echo
echo "=== Build from /tmp copy ==="
rm -rf /tmp/dgr-build
cp -r /mnt/g/Dev/Matrix-3D/submodules/diff-gaussian-rasterization-w-pose /tmp/dgr-build
cd /tmp/dgr-build
pip install --no-build-isolation . > /tmp/dgr_build3.log 2>&1
echo "exit: $?"
tail -6 /tmp/dgr_build3.log
echo "=== import test ==="
python -c "import torch; import diff_gaussian_rasterization; print('  diff_gaussian_rasterization OK')" 2>&1 | tail -3
