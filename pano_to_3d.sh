#!/usr/bin/env bash
# pano_to_3d.sh — Matrix-3D pipeline: user-provided 360° panorama -> video -> 3D scene.
#
# Usage:
#   ./pano_to_3d.sh <panorama_image> [prompt] [options]
#
#   <panorama_image>  equirectangular (2:1) 360° panorama image (jpg/png)
#   [prompt]          optional text description of the scene (text condition
#                     for video generation); may be inline text or a path to
#                     a text file containing the prompt
#
# Options:
#   --low-vram        24 GB GPUs (e.g. RTX 3090): Step 2 uses the 5B model (~12 GB VRAM)
#   --vram-mgmt       alternative low-VRAM mode (~19 GB VRAM)
#   --output-dir DIR  output directory (default: output/<image basename>)
#
# Example:
#   ./pano_to_3d.sh ./data/pano.jpg "a medieval village, cobblestone streets, clear blue sky"
#   ./pano_to_3d.sh ./data/pano.jpg ./data/pano.txt --low-vram
#
# The pipeline needs the CUDA-enabled `matrix3d` conda env (WSL side); this script
# activates it automatically. Override the defaults via environment variables:
#   CONDA_HOME  (default: /home/ron/miniconda3)
#   CONDA_ENV   (default: matrix3d)
#   CUDA_HOME   (default: /home/ron/cuda-12.6)

set -e

# --- Parse args ---
LOW_VRAM=0
VRAM_MGMT=0
PANO_PATH=""
PROMPT=""
OUTPUT_DIR=""

while [ $# -gt 0 ]; do
    case "$1" in
        --low-vram|--3090) LOW_VRAM=1 ;;
        --vram-mgmt)       VRAM_MGMT=1 ;;
        --output-dir)      OUTPUT_DIR="$2"; shift ;;
        -h|--help)
            sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        -*)
            echo "Unknown option: $1" >&2
            exit 1 ;;
        *)
            if [ -z "$PANO_PATH" ]; then
                PANO_PATH="$1"
            elif [ -z "$PROMPT" ]; then
                # Prompt may be inline text or a path to a text file
                if [ -f "$1" ]; then
                    PROMPT=$(cat "$1")
                else
                    PROMPT="$1"
                fi
            else
                echo "Unexpected argument: $1" >&2
                exit 1
            fi
            ;;
    esac
    shift
done

if [ -z "$PANO_PATH" ]; then
    echo "Error: panorama image path is required." >&2
    echo "Usage: $0 <panorama_image> [prompt|prompt_file] [--low-vram|--vram-mgmt] [--output-dir DIR]" >&2
    exit 1
fi

if [ ! -f "$PANO_PATH" ]; then
    echo "Error: panorama image not found: $PANO_PATH" >&2
    exit 1
fi

# Default prompt if none given
if [ -z "$PROMPT" ]; then
    PROMPT="a high quality 360 degree panorama of a detailed scene"
fi

# Default output dir: output/<image basename without extension>
if [ -z "$OUTPUT_DIR" ]; then
    BASENAME=$(basename "$PANO_PATH")
    BASENAME="${BASENAME%.*}"
    OUTPUT_DIR="output/${BASENAME}"
fi

mkdir -p "$OUTPUT_DIR"

# Prepare inputs: Step 2 expects pano_img.jpg (or pano_img.png) + prompt.txt in inout_dir
case "$PANO_PATH" in
    *.png|*.PNG) PANO_NAME="pano_img.png" ;;
    *)           PANO_NAME="pano_img.jpg" ;;
esac
cp "$PANO_PATH" "$OUTPUT_DIR/$PANO_NAME"
printf '%s' "$PROMPT" > "$OUTPUT_DIR/prompt.txt"

# --- Activate the conda environment (same as wsl_generate.sh) ---
CONDA_HOME="${CONDA_HOME:-/home/ron/miniconda3}"
CONDA_ENV="${CONDA_ENV:-matrix3d}"

if [ ! -f "$CONDA_HOME/etc/profile.d/conda.sh" ]; then
    echo "ERROR: conda not found at $CONDA_HOME" >&2
    echo "       Set CONDA_HOME to your Miniconda/Anaconda install path, e.g.:" >&2
    echo "       CONDA_HOME=/home/ron/miniconda3 $0 ..." >&2
    exit 1
fi

# shellcheck disable=SC1091
source "$CONDA_HOME/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV" || { echo "ERROR: failed to activate conda env '$CONDA_ENV'" >&2; exit 1; }

# CUDA runtime environment (harmless at runtime; helps if a lib is resolved at load time)
CUDA_HOME="${CUDA_HOME:-/home/ron/cuda-12.6}"
if [ -d "$CUDA_HOME" ]; then
    export CUDA_HOME
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
fi

# Sanity check: torch + CUDA must be available before the (long) pipeline runs
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('torch:', torch.__version__, '| CUDA available: True'); print('GPU  :', torch.cuda.get_device_name(0))" || {
    echo "ERROR: torch CUDA is not available in this environment. Aborting." >&2
    exit 1
}

# Step2: panorama image to video generation
# Default needs ~60 GB VRAM. On a 24 GB GPU (e.g. RTX 3090) run with
# --low-vram (5B model, ~12 GB) or --vram-mgmt (~19 GB).
VISIBLE_GPU_NUM=1
STEP2_EXTRA_ARGS=""
if [ "$LOW_VRAM" -eq 1 ]; then
    STEP2_EXTRA_ARGS="$STEP2_EXTRA_ARGS --use_5b_model"
fi
if [ "$VRAM_MGMT" -eq 1 ]; then
    STEP2_EXTRA_ARGS="$STEP2_EXTRA_ARGS --enable_vram_management"
fi
torchrun --nproc_per_node ${VISIBLE_GPU_NUM} code/panoramic_image_to_video.py \
  --inout_dir=$OUTPUT_DIR \
  --resolution=720 \
  $STEP2_EXTRA_ARGS

# Step3: 3d scene extraction
python code/panoramic_video_to_3DScene.py \
    --inout_dir=$OUTPUT_DIR \
    --resolution=720

echo "Done. 3D model: $OUTPUT_DIR/generated_3dgs_opt.ply"
