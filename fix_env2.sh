#!/usr/bin/env bash
# fix_env2.sh - Second fix pass.
# Root causes fixed here:
#   1. conda env had a CUDA 13.3 toolkit (targets/ dir) -> extensions linked libcudart.so.13
#   2. conda env had gcc 15.2 -> CUDA 12.6 nvcc rejects gcc > 13
#   3. pip build isolation hid torch from CUDA-extension builds
#   4. setuptools 83 removed pkg_resources (broke clip, pytorch_lightning, torchmetrics, diffsynth)
#   5. taming-transformers installed as an empty wheel
# Strategy: remove the conflicting FILES (not conda packages) + force system gcc 13.3 +
#           build with --no-build-isolation. Safe and fast.
set -o pipefail
LOG=/tmp/matrix3d_fix2.log
exec > >(tee "$LOG") 2>&1

echo "============================================"
echo " Matrix-3D Environment Fix (pass 2)"
echo "$(date)"
echo "============================================"

source /home/ron/miniconda3/etc/profile.d/conda.sh
conda activate matrix3d
ENV=/home/ron/miniconda3/envs/matrix3d

# --- Step 1: Remove conflicting CUDA 13.3 + gcc 15.2 files --------------------
echo
echo "=== Removing conda CUDA 13.3 targets + gcc activate scripts ==="
rm -rf "$ENV/targets"
rm -f "$ENV/etc/conda/activate.d/~cuda-nvcc_activate.sh"
rm -f "$ENV/etc/conda/activate.d/activate-gcc_linux-64.sh"
rm -f "$ENV/etc/conda/activate.d/activate-gxx_linux-64.sh"
rm -f "$ENV/etc/conda/activate.d/activate-binutils_linux-64.sh"
# Remove conda gcc/g++ binaries + symlinks so they can't shadow system gcc
rm -f "$ENV"/bin/x86_64-conda-linux-gnu-cc \
      "$ENV"/bin/x86_64-conda-linux-gnu-c++ \
      "$ENV"/bin/x86_64-conda-linux-gnu-gcc \
      "$ENV"/bin/x86_64-conda-linux-gnu-g++ \
      "$ENV"/bin/gcc "$ENV"/bin/g++ "$ENV"/bin/cc "$ENV"/bin/c++ 2>/dev/null
echo "Removed: $(ls -d "$ENV/targets" 2>/dev/null || echo 'targets/ (gone)')"

# Re-activate to drop the old activate-script env vars
conda deactivate
conda activate matrix3d

# --- Step 2: System CUDA 12.6 + system gcc 13.3 --------------------------------
export CUDA_HOME=/home/ron/cuda-12.6
export PATH=/home/ron/cuda-12.6/bin:$PATH
export LD_LIBRARY_PATH=/home/ron/cuda-12.6/lib64:$LD_LIBRARY_PATH
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
unset CUDA_CFLAGS CUDA_LDFLAGS

echo
echo "=== Build environment ==="
echo "nvcc: $(which nvcc)  ($(nvcc --version | grep -oP 'release \K[0-9.]+'))"
echo "gcc:  $(which gcc)  ($(gcc --version | head -1))"
echo "c++:  $(which c++)"

cd /mnt/g/Dev/Matrix-3D

# --- Step 3: Rebuild CUDA extensions (no build isolation) ----------------------
echo
echo "=== [3a] simple-knn ==="
cd submodules/simple-knn
rm -rf build
python setup.py install 2>&1 | tail -4
python -c "import torch; from simple_knn._C import distCUDA2; print('  simple_knn._C OK')" 2>&1 | tail -1

echo
echo "=== [3b] diff-gaussian-rasterization-w-pose ==="
cd /mnt/g/Dev/Matrix-3D
pip install --no-build-isolation git+https://github.com/rmurai0610/diff-gaussian-rasterization-w-pose.git 2>&1 | tail -4
python -c "import torch; import diff_gaussian_rasterization; print('  diff_gaussian_rasterization OK')" 2>&1 | tail -1

echo
echo "=== [3c] ODGS odgs_gaussian_rasterization ==="
cd /mnt/g/Dev/Matrix-3D/submodules/ODGS
pip install --no-build-isolation submodules/odgs-gaussian-rasterization 2>&1 | tail -4
python -c "import torch; import odgs_gaussian_rasterization; print('  odgs_gaussian_rasterization OK')" 2>&1 | tail -1

echo
echo "=== [3d] pytorch3d v0.7.7 (long compile) ==="
cd /mnt/g/Dev/Matrix-3D
pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.7" 2>&1 | tail -4
python -c "import torch; import pytorch3d; print('  pytorch3d OK', pytorch3d.__version__)" 2>&1 | tail -1

# --- Step 4: Restore pkg_resources (setuptools 83 removed it) ------------------
echo
echo "=== [4] Downgrade setuptools (<81) to restore pkg_resources ==="
pip install "setuptools<81" 2>&1 | tail -3
python -c "import pkg_resources; print('  pkg_resources OK')" 2>&1 | tail -1

# --- Step 5: Reinstall taming-transformers from a local clone ------------------
echo
echo "=== [5] Reinstall taming-transformers (local clone, no build isolation) ==="
pip uninstall -y taming-transformers 2>&1 | tail -1
rm -rf /tmp/taming-transformers
git clone --depth 1 https://github.com/CompVis/taming-transformers.git /tmp/taming-transformers 2>&1 | tail -2
cd /tmp/taming-transformers
# taming/ has no __init__.py, so find_packages() yields an empty wheel.
# Patch setup.py to use find_namespace_packages() so the namespace pkg installs.
sed -i 's/from setuptools import setup, find_packages/from setuptools import setup, find_namespace_packages as find_packages/' setup.py
pip install --no-build-isolation . 2>&1 | tail -4
python -c "import taming; import taming.modules.vqvae.quantize; print('  taming OK')" 2>&1 | tail -1

# --- Step 6: Final verification ------------------------------------------------
echo
echo "=== [6] Final verification ==="
cd /mnt/g/Dev/Matrix-3D
python /mnt/g/Dev/Matrix-3D/verify_env.py 2>&1 | tail -60

echo
echo "============================================"
echo " DONE: $(date)"
echo " Log: $LOG"
echo "============================================"
