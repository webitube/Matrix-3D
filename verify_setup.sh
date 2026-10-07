#!/usr/bin/env bash
# verify_setup.sh - Verify the Matrix-3D environment configuration and installation status.
#
# Run inside WSL:
#   wsl -- bash /mnt/g/Dev/Matrix-3D/verify_setup.sh
#
# Exit code: 0 if every check passes, 1 otherwise.
# This script is read-only: it never installs or modifies anything.

set -o pipefail

PROJECT=/mnt/g/Dev/Matrix-3D
CUDA_HOME=/home/ron/cuda-12.6
CONDA=/home/ron/miniconda3
ENV_NAME=matrix3d

PASS=0
FAIL=0
WARN=0

pass() { echo "  [PASS] $1"; PASS=$((PASS+1)); }
fail() { echo "  [FAIL] $1"; FAIL=$((FAIL+1)); }
warn() { echo "  [WARN] $1"; WARN=$((WARN+1)); }
section() { echo; echo "=== $1 ==="; }

# --- Activate the conda environment -----------------------------------------
source "$CONDA/etc/profile.d/conda.sh" 2>/dev/null
if ! conda activate "$ENV_NAME" 2>/dev/null; then
    echo "FATAL: cannot activate conda env '$ENV_NAME' (is it installed?)"
    exit 1
fi

# Put the CUDA toolkit on PATH/LD_LIBRARY_PATH so CUDA extensions can load.
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

# --- 1. Conda environment ----------------------------------------------------
section "1. Conda environment ($ENV_NAME)"
PYVER=$(python --version 2>&1)
echo "  python: $PYVER"
echo "  path:   $(which python)"
case "$PYVER" in
    *3.10*) pass "Python 3.10" ;;
    *)      fail "Python is $PYVER (expected 3.10.x)" ;;
esac

# --- 2. CUDA toolkit ---------------------------------------------------------
section "2. CUDA toolkit ($CUDA_HOME)"
if [ -x "$CUDA_HOME/bin/nvcc" ]; then
    NVVER=$("$CUDA_HOME/bin/nvcc" --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+')
    if [ "$NVVER" = "12.6" ]; then
        pass "nvcc $NVVER"
    else
        fail "nvcc version is ${NVVER:-unknown} (expected 12.6)"
    fi
else
    fail "nvcc not found at $CUDA_HOME/bin/nvcc"
fi
[ -e "$CUDA_HOME/lib64/libcudart.so" ] && pass "libcudart.so present" || fail "libcudart.so missing"
[ -e "$CUDA_HOME/include/cuda_runtime.h" ] && pass "cuda_runtime.h header present" || fail "cuda_runtime.h header missing"

# --- 3. GPU ------------------------------------------------------------------
section "3. GPU"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null | sed 's/^/  /'
    pass "nvidia-smi works"
else
    fail "nvidia-smi not found (no GPU driver visible in WSL)"
fi

# --- 4. PyTorch CUDA ---------------------------------------------------------
section "4. PyTorch CUDA"
TORCH_CUDA=$(python - <<'EOF' 2>&1
import torch
print(torch.__version__)
print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
    print(torch.version.cuda)
EOF
)
echo "$TORCH_CUDA" | sed 's/^/  /'
if echo "$TORCH_CUDA" | grep -q "True"; then
    pass "torch.cuda.is_available() == True"
else
    fail "torch.cuda.is_available() is False"
fi

# --- 5. Pinned package versions ---------------------------------------------
section "5. Pinned package versions"
check_version() {
    local pkg="$1" expected="$2" actual
    actual=$(python -c "import importlib.metadata as m; print(m.version('$pkg'))" 2>/dev/null)
    if [ -z "$actual" ]; then
        fail "$pkg: not installed"
    elif [ "$actual" = "$expected" ]; then
        pass "$pkg==$expected"
    else
        fail "$pkg: expected $expected, got $actual"
    fi
}
check_version torch 2.7.1
check_version torchvision 0.22.1
check_version xformers 0.0.31
check_version transformers 4.56.0
check_version flash-attn 2.7.4.post1
check_version pytorch-lightning 1.4.2
# streamlit is intentionally NOT installed: it pins protobuf<4, which
# conflicts with tensorboard's protobuf>=6.31.1. Only the optional
# DiffSynth web app needs it.

# --- 6. Python import smoke test --------------------------------------------
section "6. Python import smoke test (verify_env.py)"
VERIFY_OUT=$(python "$PROJECT/verify_env.py" 2>&1)
VERIFY_RC=$?
echo "$VERIFY_OUT" | tail -n 45
if [ "$VERIFY_RC" -eq 0 ]; then
    pass "all modules import cleanly"
else
    fail "some modules failed to import (see above)"
fi

# --- 7. Checkpoints ----------------------------------------------------------
section "7. Checkpoints"
CHECKPOINTS=(
    "checkpoints/moge/model.pt"
    "checkpoints/StableSR/stablesr_turbo.ckpt"
    "checkpoints/StableSR/vqgan_cfw_00011.ckpt"
    "checkpoints/flux_lora/checkpoints/text2panoimage_lora.safetensors"
    "checkpoints/pano_lrm/checkpoints/pano_lrm_480p.pt"
    "checkpoints/Wan-AI/wan_lora/checkpoints/pano_video_gen_480p.ckpt"
    "checkpoints/Wan-AI/wan_lora/checkpoints/pano_video_gen_720p.bin"
    "checkpoints/Wan-AI/wan_lora/checkpoints/pano_video_gen_720p_5b.safetensors"
)
for c in "${CHECKPOINTS[@]}"; do
    if [ -s "$PROJECT/$c" ]; then
        pass "$c ($(du -h "$PROJECT/$c" | cut -f1))"
    else
        fail "$c (missing or empty)"
    fi
done
# Optional VEnhancer checkpoint (only needed for the VideoSR enhancement step).
if [ -s "$PROJECT/code/VideoSR/checkpoints/venhancer_v2.pt" ]; then
    pass "code/VideoSR/checkpoints/venhancer_v2.pt (optional)"
else
    warn "code/VideoSR/checkpoints/venhancer_v2.pt missing (optional, VideoSR only)"
fi

# --- 8. Submodules -----------------------------------------------------------
section "8. Submodules"
for s in nvdiffrast ODGS simple-knn; do
    if [ -d "$PROJECT/submodules/$s" ] && [ -n "$(ls -A "$PROJECT/submodules/$s" 2>/dev/null)" ]; then
        pass "submodules/$s"
    else
        fail "submodules/$s (missing or empty)"
    fi
done

# --- Summary -----------------------------------------------------------------
section "Summary"
echo "  PASS: $PASS   FAIL: $FAIL   WARN: $WARN"
if [ "$FAIL" -gt 0 ]; then
    echo "  RESULT: FAIL"
    exit 1
else
    echo "  RESULT: PASS"
    exit 0
fi
