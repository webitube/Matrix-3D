#!/bin/bash
# ============================================================
# Matrix-3D Environment Fix Script (WSL2)
# Order matters: upgrade torch FIRST, then build CUDA exts
# ============================================================
set -o pipefail

LOG=/tmp/matrix3d_fix.log
exec > >(tee "$LOG") 2>&1

echo "============================================"
echo " Matrix-3D Environment Fix"
echo " $(date)"
echo "============================================"

# --- Activate conda env ---
source /home/ron/miniconda3/etc/profile.d/conda.sh
conda activate matrix3d
echo "Python: $(python --version)"
echo "pip: $(pip --version)"

# --- Set up CUDA environment ---
export CUDA_HOME=/home/ron/cuda-12.6
export PATH=/home/ron/cuda-12.6/bin:$PATH
export LD_LIBRARY_PATH=/home/ron/cuda-12.6/lib64:$LD_LIBRARY_PATH
echo "nvcc: $(which nvcc)"
nvcc --version | tail -1
echo "CUDA_HOME=$CUDA_HOME"

# ============================================================
# PHASE 0: Upgrade torch to 2.7.1 (xformers 0.0.31 requires it)
# Do this FIRST so all CUDA extensions build against 2.7.1
# ============================================================
echo ""
echo "============================================"
echo " PHASE 0: Upgrade torch to 2.7.1"
echo "============================================"
pip install torch==2.7.1 torchvision==0.22.1 2>&1 | tail -5
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# ============================================================
# PHASE 1: Install missing pip packages
# ============================================================
echo ""
echo "============================================"
echo " PHASE 1: Missing pip packages"
echo "============================================"

echo "--- Installing timm, lpips, onnx ---"
pip install timm lpips onnx 2>&1 | tail -5

echo "--- Installing pytorch_lightning ---"
pip install pytorch-lightning==1.4.2 2>&1 | tail -5

echo "--- Installing openai-clip (fixes 'clip' import) ---"
pip install openai-clip 2>&1 | tail -5

echo "--- Installing basicsr (for StableSR/realesrgan) ---"
pip install basicsr 2>&1 | tail -5

echo "--- Fixing streamlit (altair.vegalite.v4) ---"
pip install streamlit==1.12.1 2>&1 | tail -5

# ============================================================
# PHASE 2: Rebuild CUDA extensions (against torch 2.7.1)
# ============================================================
echo ""
echo "============================================"
echo " PHASE 2: CUDA extensions"
echo "============================================"

cd /mnt/g/Dev/Matrix-3D

# --- 2a: nvdiffrast ---
echo ""
echo "--- [2a] nvdiffrast ---"
cd /mnt/g/Dev/Matrix-3D/submodules/nvdiffrast/
pip install . 2>&1 | tail -10
cd /mnt/g/Dev/Matrix-3D

# --- 2b: simple-knn ---
echo ""
echo "--- [2b] simple-knn ---"
cd /mnt/g/Dev/Matrix-3D/submodules/simple-knn/
python setup.py install 2>&1 | tail -10
cd /mnt/g/Dev/Matrix-3D

# --- 2c: diff-gaussian-rasterization-w-pose ---
echo ""
echo "--- [2c] diff-gaussian-rasterization-w-pose ---"
pip install git+https://github.com/rmurai0610/diff-gaussian-rasterization-w-pose.git 2>&1 | tail -10

# --- 2d: ODGS (odgs_gaussian_rasterization) ---
echo ""
echo "--- [2d] ODGS odgs_gaussian_rasterization ---"
cd /mnt/g/Dev/Matrix-3D/submodules/ODGS
pip install submodules/odgs-gaussian-rasterization 2>&1 | tail -10
cd /mnt/g/Dev/Matrix-3D

# --- 2e: flash-attn ---
echo ""
echo "--- [2e] flash-attn 2.7.4.post1 ---"
pip install flash-attn==2.7.4.post1 --no-build-isolation 2>&1 | tail -10

# --- 2f: pytorch3d ---
echo ""
echo "--- [2f] pytorch3d v0.7.7 ---"
pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.7" 2>&1 | tail -10

# --- 2g: DiffSynth-Studio ---
echo ""
echo "--- [2g] DiffSynth-Studio ---"
cd /mnt/g/Dev/Matrix-3D/code/DiffSynth-Studio/
pip install -e . 2>&1 | tail -10
cd /mnt/g/Dev/Matrix-3D

# --- 2h: taming-transformers (without #egg fragment) ---
echo ""
echo "--- [2h] taming-transformers ---"
pip install git+https://github.com/CompVis/taming-transformers.git@master 2>&1 | tail -10

# ============================================================
# PHASE 3: Fix remaining dependency conflicts
# ============================================================
echo ""
echo "============================================"
echo " PHASE 3: Fix dependency conflicts"
echo "============================================"

echo "--- Re-pin transformers to 4.56.0 ---"
pip install transformers==4.56.0 2>&1 | tail -5

echo "--- Fix protobuf conflict ---"
pip install protobuf 2>&1 | tail -5

# ============================================================
# PHASE 4: Verify
# ============================================================
echo ""
echo "============================================"
echo " PHASE 4: Verification"
echo "============================================"
python /mnt/g/Dev/Matrix-3D/verify_env.py 2>&1

echo ""
echo "============================================"
echo " DONE: $(date)"
echo " Log: $LOG"
echo "============================================"
