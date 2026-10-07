#!/usr/bin/env bash
# One-shot WSL2 setup for Matrix-3D (no sudo required).
# Installs Miniconda + CUDA 12.4 toolkit (user prefix) + matrix3d env (py3.10, torch cu124).
set -uo pipefail
export DEBIAN_FRONTEND=noninteractive

HOME_DIR="/home/ron"
MC="$HOME_DIR/miniconda3"
CUDA_PREFIX="$HOME_DIR/cuda-12.4"
CUDA_RUN="cuda_12.4.0_550.54.14_linux.run"

echo "=== [1/4] Clean up bad install ==="
rm -rf "/tmp/C:Userswebit" 2>/dev/null || true

echo "=== [2/4] Install Miniconda to $MC ==="
if [ -x "$MC/bin/conda" ]; then
  echo "Miniconda already present"
else
  cd /tmp
  [ -f Miniconda3-latest-Linux-x86_64.sh ] || wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash Miniconda3-latest-Linux-x86_64.sh -b -p "$MC"
fi
"$MC/bin/conda" --version
"$MC/bin/conda" init bash >/dev/null 2>&1 || true

echo "=== [3/4] Install CUDA 12.4 toolkit (no sudo) to $CUDA_PREFIX ==="
if [ -x "$CUDA_PREFIX/bin/nvcc" ]; then
  echo "CUDA toolkit already present"
else
  cd /tmp
  if [ ! -f "$CUDA_RUN" ]; then
    echo "Downloading CUDA 12.4 runfile (~4.5GB)..."
    wget -q "https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/$CUDA_RUN"
  fi
  sh "$CUDA_RUN" --toolkit --silent --no-drm --prefix="$CUDA_PREFIX"
fi
"$CUDA_PREFIX/bin/nvcc" --version

echo "=== [4/4] Create matrix3d env + torch cu124 ==="
source "$MC/etc/profile.d/conda.sh"
conda create -y -n matrix3d python=3.10
conda activate matrix3d
export PATH="$CUDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_PREFIX/lib64:${LD_LIBRARY_PATH:-}"
pip install --upgrade pip
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), (torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''))"

echo "=== SETUP DONE ==="
