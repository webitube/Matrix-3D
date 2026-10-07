#!/usr/bin/env bash
# Fix pass: inspect CUDA runfile options, accept conda ToS, create env, install torch cu124.
set -uo pipefail
export DEBIAN_FRONTEND=noninteractive
HOME_DIR="/home/ron"
MC="$HOME_DIR/miniconda3"
CUDA_RUN="/tmp/cuda_12.4.0_550.54.14_linux.run"

echo "=== [1] runfile info ==="
ls -la "$CUDA_RUN"
head -c 200 "$CUDA_RUN" | od -c | head -5
echo "--- runfile --help (options) ---"
sh "$CUDA_RUN" --help 2>&1 | head -80

echo "=== [2] conda ToS accept ==="
source "$MC/etc/profile.d/conda.sh"
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

echo "=== [3] create matrix3d env ==="
conda create -y -n matrix3d python=3.10
conda activate matrix3d
python --version
pip install --upgrade pip

echo "=== [4] install torch cu124 ==="
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), (torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''))"

echo "=== FIX PASS DONE ==="
