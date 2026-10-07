#!/usr/bin/env bash
# generate.sh — Matrix-3D pipeline: text/image -> panorama -> video -> 3D scene.
#
# Usage:
#   ./generate.sh               # default (Step 2 needs ~60 GB VRAM)
#   ./generate.sh --low-vram    # 24 GB GPUs (e.g. RTX 3090): Step 2 uses the
#                               #   5B model (~12 GB VRAM)
#   ./generate.sh --vram-mgmt   # alternative low-VRAM mode (~19 GB VRAM)

# --- Parse low-VRAM switches (for 24 GB GPUs, e.g. RTX 3090) ---
LOW_VRAM=0
VRAM_MGMT=0
for arg in "$@"; do
    case "$arg" in
        --low-vram|--3090) LOW_VRAM=1 ;;
        --vram-mgmt)       VRAM_MGMT=1 ;;
    esac
done

output_dir=output/example1

# Step1: text to panorama image
python code/panoramic_image_generation.py \
    --mode=t2p \
    --prompt="a medieval village, half-timbered houses, cobblestone streets, lush greenery, clear blue sky, detailed textures, vibrant colors, high resolution" \
    --output_path=$output_dir

# Or you can choose image to panorama image generation
# python code/panoramic_image_generation.py \
#     --mode=i2p \
#     --input_image_path="./data/image2.jpg" \
#     --output_path=$output_dir

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
  --inout_dir=$output_dir  \
  --resolution=720 \
  $STEP2_EXTRA_ARGS

# Step3: 3d scene extraction
python code/panoramic_video_to_3DScene.py \
    --inout_dir=$output_dir \
    --resolution=720
