# AGENTS.md — Matrix-3D

Instructions for AI coding agents working in this repository.
For full structural context, read `ARCHITECTURE.md` first.

## Project Overview

Matrix-3D generates omnidirectional explorable 3D worlds from text or a single image.
Three-stage pipeline:

1. **Step 1** — text/image → 360° panorama (`code/panoramic_image_generation.py`, FLUX.1 + LoRA)
2. **Step 2** — panorama → MoGe depth → camera rail render → Wan video (`code/panoramic_image_to_video.py`)
3. **Step 3** — video → perspective depth → datagen → StableSR → 3D Gaussian Splatting (`code/panoramic_video_to_3DScene.py`)

Stack: Python 3 + PyTorch (CUDA), nvdiffrast, OpenCV (video I/O — **no ffmpeg**), Gradio, bash orchestrators.
Heavy models are vendored under `code/`; weights live in `checkpoints/`.

## Commands

**All pipeline commands run in WSL, in the conda env `matrix3d` — never in the Windows base env.**

```bash
# Activate env (prefix for any Python command in WSL):
wsl bash -lc "source /home/ron/miniconda3/etc/profile.d/conda.sh && conda activate matrix3d && cd /mnt/d/Dev/Matrix-3D && <cmd>"

# Syntax-check a Python file (do this after every edit):
wsl bash -lc "source /home/ron/miniconda3/etc/profile.d/conda.sh && conda activate matrix3d && cd /mnt/d/Dev/Matrix-3D && python -m py_compile code/panoramic_image_to_video.py"

# Bash script syntax check:
wsl bash -lc "bash -n /mnt/d/Dev/Matrix-3D/pano_to_3d.sh"

# Run the pipeline (panorama -> video -> 3D scene), 24 GB GPU:
wsl -- bash /mnt/d/Dev/Matrix-3D/pano_to_3d.sh <pano.png> <prompt.txt> --low-vram

# Full pipeline incl. panorama generation:
wsl -- bash /mnt/d/Dev/Matrix-3D/wsl_generate.sh --low-vram

# Download checkpoints:
wsl bash -lc "source /home/ron/miniconda3/etc/profile.d/conda.sh && conda activate matrix3d && cd /mnt/d/Dev/Matrix-3D && python code/download_checkpoints.py"
```

There is **no test suite**. Verify changes with `py_compile` / `bash -n` and, when feasible, a short pipeline run.

## Code Style

- **Python**: follow the existing style — 4-space indent, snake_case functions, `os.path.join` for paths, `print()` for progress logging (no logging framework in first-party scripts). Keep edits minimal and surgical; the codebase mixes research code and glue code.
- **Bash**: orchestrator scripts use `set -e`, `--opt value` and `--opt=value` passthrough patterns. When adding a new Step-2 CLI flag, add it to **both** `argparse` in `code/panoramic_image_to_video.py` (with underscore AND hyphen aliases, e.g. `--liss_segments`, `--liss-segments`) **and** `STEP2_VALUE_OPTS` in `pano_to_3d.sh` (value-taking options only). Update the `sed -n '2,NNp'` help range in `pano_to_3d.sh` if the header comment grows.
- **No new top-level dependencies** without checking the `matrix3d` env has them.
- **Video I/O**: use `write_video` / `get_video_frames` / `concat_videos_streaming` from `code/utils_3dscene/pipeline_utils_3dscene.py` (cv2-based). Do not introduce ffmpeg calls — ffmpeg is not installed in WSL.
- **Frame counts**: any video frame count must satisfy `num_frames % 4 == 1` (Wan VAE constraint).
- **Frame rates**: generated video = 24 fps; condition/rendered videos = 12 fps. Do not mix.

## Boundaries & Constraints

### Do NOT modify (vendored black boxes)

- `code/DiffSynth-Studio/`, `code/MoGe/`, `code/StableSR/`, `code/VideoSR/`, `code/Pano_GS_Opt/`, `code/Pano_LRM/`, `code/pano_init/`
  - Small Exception: `code/Pano_GS_Opt/train.py`: Small modifications to add memory and progress logging.
- `submodules/` (CUDA C++ extensions: nvdiffrast, simple-knn, diff-gaussian-rasterization-w-pose, ODGS)
- `checkpoints/`, `models/`, `output/`, `data/` (weights and artifacts, not source)

Interact with vendored packages only via `sys.path.append` + import (DiffSynth-Studio, Pano_LRM, pano_init) or `os.system("cd code/<pkg> && python <script> ...")` (MoGe, StableSR, VideoSR, Pano_GS_Opt).

### Circular import rule (CRITICAL)

`code/utils_3dscene/pipeline_utils_3dscene.py` imports from `nvrender.py` at module top level.
Therefore `nvrender.py` must **never** import from `pipeline_utils_3dscene` at top level — use a local import inside the function body (see `render_rts_segmented`).

### Pipeline contract

Steps communicate only through files in the `inout_dir` (see `ARCHITECTURE.md` §3.4). When changing what Step 2 writes or Step 3 reads, update both sides and the layout section of `ARCHITECTURE.md`.

- `condition/cameras.npz` must always contain the **full** camera rail (one 4×4 Rt per generated video frame), even when Option A segmentation is used.
- Step 3 indexes cameras by global frame index (`int(exr_filename.split(".")[0])`), so per-segment camera files are not acceptable.

### Memory constraints (RTX 3090, 24 GB VRAM / 96 GB RAM)

- Never allocate a `(total_frames, H, W, 3)` GPU tensor for long rails — use the segmented path (`render_rts_segmented`, Option A) which keeps per-segment tensors ~81 frames.
- Wan 14B needs `--vram-mgmt` (~19 GB); Wan 5B is the `--low-vram` default (~12 GB).
- FLUX uses `enable_sequential_cpu_offload()` — do not replace with `.to(device)`.

### Environment

- Windows `python`/`conda` are NOT on PATH and are the wrong env. Always go through WSL + conda `matrix3d`.
- `CUDA_HOME` defaults to `/home/ron/cuda-12.6` inside WSL.
- `OPENCV_IO_ENABLE_OPENEXR=1` is set by the scripts; keep it when adding scripts that read EXR.
