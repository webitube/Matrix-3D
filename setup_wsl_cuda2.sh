#!/usr/bin/env bash
# Install CUDA 12.6.2 toolkit (no root, user prefix) then run Matrix-3D install.sh.
set -o pipefail
export DEBIAN_FRONTEND=noninteractive
MC="/home/ron/miniconda3"
CUDA_PREFIX="/home/ron/cuda-12.6"
RUN="/tmp/cuda_12.6.2_560.35.03_linux.run"
URL="https://developer.download.nvidia.com/compute/cuda/12.6.2/local_installers/cuda_12.6.2_560.35.03_linux.run"
REPO="/mnt/g/Dev/Matrix-3D"

source "$MC/etc/profile.d/conda.sh"
conda activate matrix3d

echo "=== [0] remove buggy conda cuda-nvcc 13.3 packages ==="
conda remove -y --force -n matrix3d cuda-nvcc cuda-toolkit cuda-compiler cuda-command-line-tools cuda-libraries cuda-libraries-dev cuda-tools cuda-visual-tools cuda-nsight cuda-nvprof cuda-nvvp 2>&1 | tail -5
conda activate matrix3d

echo "=== [1] CUDA 12.6.2 toolkit -> $CUDA_PREFIX (no root) ==="
if [ -x "$CUDA_PREFIX/bin/nvcc" ]; then
  echo "already present"
else
  cd /tmp
  if [ ! -s "$RUN" ]; then
    echo "verifying URL..."
    wget --spider "$URL" 2>&1 | tail -3
    echo "downloading runfile (~4.9GB)..."
    wget --tries=3 --timeout=120 -O "$RUN" "$URL"
  fi
  SZ=$(stat -c%s "$RUN" 2>/dev/null || echo 0)
  echo "runfile size: $SZ bytes"
  if [ "$SZ" -lt 4000000000 ]; then
    echo "ERROR: runfile too small, download failed"; exit 1
  fi
  sh "$RUN" --toolkit --silent --no-drm --prefix="$CUDA_PREFIX"
fi
export PATH="$CUDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_PREFIX/lib64:${LD_LIBRARY_PATH:-}"
echo "--- nvcc ---"; nvcc --version

echo "=== [2] verify torch GPU ==="
python -c "import torch; print('cuda_avail', torch.cuda.is_available(), torch.cuda.get_device_name(0)); x=torch.randn(4,4,device='cuda'); print('gpu matmul ok', float((x@x).sum()))"

echo "=== [3] run install.sh (continues past failures) ==="
cd "$REPO"
bash install.sh 2>&1 | tee /tmp/matrix3d_install.log
echo "=== INSTALL.SH EXIT: ${PIPESTATUS[0]} ==="
