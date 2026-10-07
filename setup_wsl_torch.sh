#!/usr/bin/env bash
# Install torch (PyPI default = cu126) + CUDA 12.6 toolkit via conda (no root).
set -uo pipefail
export DEBIAN_FRONTEND=noninteractive
MC="/home/ron/miniconda3"
source "$MC/etc/profile.d/conda.sh"
conda activate matrix3d
echo "=== env ==="; which python; python --version

echo "=== [1] torch 2.7.0 (PyPI default, cu126) ==="
pip install torch==2.7.0 torchvision==0.22.0
python -c "import torch; print('torch', torch.__version__, 'cuda_built', torch.version.cuda, 'cuda_avail', torch.cuda.is_available())"

echo "=== [2] CUDA 12.6 toolkit via conda (nvidia channel) ==="
conda install -y -c nvidia cuda-toolkit=12.6
echo "--- nvcc ---"
which nvcc
nvcc --version

echo "=== [3] verify torch sees GPU ==="
python -c "import torch; print('cuda_avail', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'); x=torch.randn(3,3,device='cuda'); print('matmul ok', (x@x).sum().item() is not None)"

echo "=== TORCH+CUDA DONE ==="
