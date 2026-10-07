#!/bin/bash
# ============================================================
# Reinstall CUDA 12.6.2 toolkit (no-root) + remove conda nvcc
# ============================================================
set -o pipefail

LOG=/tmp/matrix3d_cuda_reinstall.log
exec > >(tee "$LOG") 2>&1

echo "============================================"
echo " CUDA 12.6.2 Reinstall"
echo " $(date)"
echo "============================================"

# --- Step 1: Remove buggy conda nvcc 13.3 from the env ---
echo ""
echo "--- Removing conda nvcc 13.3 from matrix3d env ---"
ENV=/home/ron/miniconda3/envs/matrix3d
rm -f $ENV/bin/nvcc $ENV/bin/nvcc* 2>/dev/null
rm -rf $ENV/include/cuda* $ENV/include/thrust $ENV/include/crt 2>/dev/null
rm -rf $ENV/lib/libcudart* $ENV/lib/libcublas* $ENV/lib/libcufft* \
       $ENV/lib/libcurand* $ENV/lib/libcusolver* $ENV/lib/libcusparse* \
       $ENV/lib/libnvToolsExt* $ENV/lib/libnvrtc* $ENV/lib/libnvJitLink* \
       $ENV/lib/libcudadevrt* $ENV/lib/libcudart_static* 2>/dev/null
rm -rf $ENV/lib64/libcudart* $ENV/lib64/libcublas* $ENV/lib64/libcufft* \
       $ENV/lib64/libcurand* $ENV/lib64/libcusolver* $ENV/lib64/libcusparse* \
       $ENV/lib64/libnvToolsExt* $ENV/lib64/libnvrtc* $ENV/lib64/libnvJitLink* \
       $ENV/lib64/libcudadevrt* $ENV/lib64/libcudart_static* 2>/dev/null
echo "nvcc after removal: $(ls $ENV/bin/nvcc 2>/dev/null || echo 'GONE (good)')"

# --- Step 2: Download CUDA 12.6.2 runfile ---
echo ""
echo "--- Downloading CUDA 12.6.2 runfile (4.1 GB) ---"
RUNFILE=/tmp/cuda_12.6.2_560.35.03_linux.run
if [ ! -f "$RUNFILE" ]; then
    wget -q --show-progress \
        "https://developer.download.nvidia.com/compute/cuda/12.6.2/local_installers/cuda_12.6.2_560.35.03_linux.run" \
        -O "$RUNFILE"
    if [ $? -ne 0 ]; then
        echo "ERROR: Download failed"
        exit 1
    fi
else
    echo "Runfile already exists: $RUNFILE"
fi
ls -lh "$RUNFILE"

# --- Step 3: Install CUDA toolkit (no-root, prefix) ---
echo ""
echo "--- Installing CUDA 12.6.2 toolkit to /home/ron/cuda-12.6 ---"
rm -rf /home/ron/cuda-12.6
sh "$RUNFILE" --toolkit --silent --no-drm --toolkitpath=/home/ron/cuda-12.6
if [ $? -ne 0 ]; then
    echo "ERROR: CUDA toolkit install failed"
    exit 1
fi

# --- Step 4: Verify ---
echo ""
echo "--- Verifying CUDA install ---"
ls /home/ron/cuda-12.6/bin/nvcc && echo "nvcc: OK"
/home/ron/cuda-12.6/bin/nvcc --version | tail -1
ls /home/ron/cuda-12.6/lib64/libcudart.so* && echo "cudart: OK"
ls /home/ron/cuda-12.6/include/cuda_runtime.h && echo "headers: OK"

echo ""
echo "============================================"
echo " CUDA REINSTALL DONE: $(date)"
echo "============================================"
