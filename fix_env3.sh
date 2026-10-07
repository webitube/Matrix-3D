#!/usr/bin/env bash
# fix_env3.sh - Third fix pass. Fixes the remaining pipeline-critical failures:
#   1. diff_gaussian_rasterization  - rebuild from local patched clone (added <cstdint>)
#   2. odgs_gaussian_rasterization  - rebuild from local patched submodule (added <cstdint>)
#   3. diffsynth                    - re-run editable install (pkg_resources now present)
#   4. pytorch_lightning + onnx     - drop streamlit (pins protobuf<4), upgrade protobuf>=6.31.1
# Non-pipeline items left as-is: streamlit (DiffSynth app only), pip basicsr/realesrgan
# (pipeline uses local code/StableSR/basicsr), open3d (eval scripts, needs system libusb).
set -o pipefail
LOG=/tmp/matrix3d_fix3.log
exec > >(tee "$LOG") 2>&1

echo "============================================"
echo " Matrix-3D Environment Fix (pass 3)"
echo "$(date)"
echo "============================================"

source /home/ron/miniconda3/etc/profile.d/conda.sh
conda activate matrix3d
export CUDA_HOME=/home/ron/cuda-12.6
export PATH=/home/ron/cuda-12.6/bin:$PATH
export LD_LIBRARY_PATH=/home/ron/cuda-12.6/lib64:$LD_LIBRARY_PATH
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
unset CUDA_CFLAGS CUDA_LDFLAGS

cd /mnt/g/Dev/Matrix-3D

echo
echo "=== [1] diff-gaussian-rasterization-w-pose (local patched clone) ==="
cd /mnt/g/Dev/Matrix-3D/submodules/diff-gaussian-rasterization-w-pose
pip install --no-build-isolation . 2>&1 | tail -4
python -c "import torch; import diff_gaussian_rasterization; print('  diff_gaussian_rasterization OK')" 2>&1 | tail -1

echo
echo "=== [2] ODGS odgs_gaussian_rasterization (local patched) ==="
cd /mnt/g/Dev/Matrix-3D/submodules/ODGS
pip install --no-build-isolation submodules/odgs-gaussian-rasterization 2>&1 | tail -4
python -c "import torch; import odgs_gaussian_rasterization; print('  odgs_gaussian_rasterization OK')" 2>&1 | tail -1

echo
echo "=== [3] DiffSynth-Studio (editable) ==="
cd /mnt/g/Dev/Matrix-3D/code/DiffSynth-Studio
pip install --no-build-isolation -e . 2>&1 | tail -4
python -c "import diffsynth; print('  diffsynth OK')" 2>&1 | tail -1

echo
echo "=== [4] Fix protobuf conflict (drop streamlit, upgrade protobuf) ==="
pip uninstall -y streamlit 2>&1 | tail -2
pip install "protobuf>=6.31.1,<7" 2>&1 | tail -3
python -c "import pytorch_lightning; print('  pytorch_lightning OK', pytorch_lightning.__version__)" 2>&1 | tail -1
python -c "import onnx; print('  onnx OK', onnx.__version__)" 2>&1 | tail -1

echo
echo "=== [5] Final verification ==="
cd /mnt/g/Dev/Matrix-3D
python /mnt/g/Dev/Matrix-3D/verify_env.py 2>&1 | grep -E "^OK|^FAIL|torch.cuda|GPU:|cudnn"

echo
echo "============================================"
echo " DONE: $(date)"
echo " Log: $LOG"
echo "============================================"
