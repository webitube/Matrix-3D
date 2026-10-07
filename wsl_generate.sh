#!/usr/bin/env bash
# wsl_generate.sh — Run ./generate.sh inside the correct WSL conda environment.
#
# The Matrix-3D pipeline needs the CUDA-enabled `matrix3d` conda env (WSL side),
# so this script activates it, verifies torch+CUDA, then runs ./generate.sh.
#
# Usage (from PowerShell):
#   wsl -- bash /mnt/g/Dev/Matrix-3D/wsl_generate.sh
#
# Usage (from inside a WSL shell):
#   bash wsl_generate.sh
#
# Any extra arguments are passed through to generate.sh, e.g.:
#   bash wsl_generate.sh --low-vram    # 24 GB GPUs (e.g. RTX 3090):
#                                      #   Step 2 uses the 5B model (~12 GB VRAM)
#   bash wsl_generate.sh --vram-mgmt   # alternative low-VRAM mode (~19 GB VRAM)
#
# Override the defaults via environment variables if your setup differs:
#   CONDA_HOME  (default: /home/ron/miniconda3)
#   CONDA_ENV   (default: matrix3d)
#   CUDA_HOME   (default: /home/ron/cuda-12.6)

# --- Locate the project root (directory containing this script) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || { echo "ERROR: cannot cd to $SCRIPT_DIR"; exit 1; }

if [ ! -f generate.sh ]; then
    echo "ERROR: generate.sh not found in $SCRIPT_DIR"
    exit 1
fi

# --- Activate the conda environment ---
CONDA_HOME="${CONDA_HOME:-/home/ron/miniconda3}"
CONDA_ENV="${CONDA_ENV:-matrix3d}"

if [ ! -f "$CONDA_HOME/etc/profile.d/conda.sh" ]; then
    echo "ERROR: conda not found at $CONDA_HOME"
    echo "       Set CONDA_HOME to your Miniconda/Anaconda install path, e.g.:"
    echo "       CONDA_HOME=/home/ron/miniconda3 bash wsl_generate.sh"
    exit 1
fi

# shellcheck disable=SC1091
source "$CONDA_HOME/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV" || { echo "ERROR: failed to activate conda env '$CONDA_ENV'"; exit 1; }

# --- CUDA runtime environment (harmless at runtime; helps if a lib is resolved at load time) ---
CUDA_HOME="${CUDA_HOME:-/home/ron/cuda-12.6}"
if [ -d "$CUDA_HOME" ]; then
    export CUDA_HOME
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
fi

# --- Sanity checks ---
echo "=== wsl_generate.sh ==="
echo "Project dir : $SCRIPT_DIR"
echo "Python      : $(command -v python)"

python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('torch       :', torch.__version__, '| CUDA available: True'); print('GPU         :', torch.cuda.get_device_name(0))" || {
    echo "ERROR: torch CUDA is not available in this environment. Aborting."
    exit 1
}

# --- Run the pipeline ---
echo "=== Running ./generate.sh ==="
./generate.sh "$@"
exit $?
