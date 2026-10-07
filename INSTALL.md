# Matrix-3D — Installation Guide

This guide covers the full installation process for **Matrix-3D: Omnidirectional Explorable 3D World Generation**. It is written for **Linux with an NVIDIA GPU** (the only supported platform). It also documents the exact steps and fixes needed to get a clean, working environment, including the CUDA extension builds that the stock `install.sh` does not handle on its own.

> **TL;DR** — clone the repo, create a `matrix3d` conda env (Python 3.10), install the CUDA 12.x toolkit, run `./install.sh`, download the checkpoints, then run `./generate.sh`.

---

## 1. Requirements

### Hardware
| Component | Minimum | Recommended |
| :-- | :-- | :-- |
| GPU (NVIDIA) | 16 GB VRAM | 24 GB VRAM (e.g. RTX 3090 / 4090) |
| RAM | 32 GB | 64 GB |
| Disk | ~40 GB free **on an SSD** | ~60 GB free **on an SSD** (code + checkpoints + CUDA toolkit) |

> **VRAM by step** (each step loads/unloads its own model, so they don't stack):
> | Step | Model | VRAM |
> | :-- | :-- | :-- |
> | 1 — Text/Image → panorama | FLUX.1-dev + LoRA (sequential CPU offload) | ~6–8 GB |
> | 2 — Panorama → video (720p) | PanoVideoGen-720p (default) | **~60 GB** |
> | 2 — Panorama → video (720p) | PanoVideoGen-720p-5B (`--use_5b_model`) | ~12 GB |
> | 2 — Panorama → video (720p) | low-VRAM mode (`--enable_vram_management`) | ~19 GB |
> | 3 — Video → 3D scene (optimization) | Pano_GS_Opt | ~10 GB |
>
> The **default** `generate.sh` (720p, no flags) peaks at **~60 GB** in Step 2. On a **24 GB** card (e.g. RTX 3090) run `./generate.sh --low-vram` (Step 2 uses the 5B model, ~12 GB) or `./generate.sh --vram-mgmt` (~19 GB).
>
> **Step 1 on a 24 GB card:** FLUX.1-dev is ~24 GB (transformer) + ~12 GB (text encoders) in bf16, so it does *not* fit on the GPU all at once. The code uses `enable_sequential_cpu_offload()`, which streams one layer at a time and keeps peak VRAM to a few GB. This is slower than keeping the model resident, but it reliably fits a 24 GB GPU. (If you have ≥ 48 GB VRAM you can switch to `enable_model_cpu_offload()` for a speedup.)

> **⚠️ Use an SSD — not a spinning hard drive (HDD).** The project (especially the `checkpoints/` and `models/` folders, which hold tens of GB of weights) **and the WSL2 Ubuntu archive** (the `ext4.vhdx` under `%LOCALAPPDATA%\Packages\...\LocalState\`) must live on an **SSD**. Every generation step streams large model weights and intermediate artifacts (panoramas, video frames, depth maps, Gaussian splats) from disk, so on an HDD the pipeline becomes **disk-IO bound and generation performance is greatly reduced** — often by an order of magnitude. If you are on WSL2, move the distro to an SSD-backed path (e.g. `wsl --export` / `wsl --import` to an SSD drive) and keep the repo + checkpoints on that same SSD.

### Software
- **OS:** Linux (Ubuntu 20.04/22.04 tested). WSL2 on Windows works.
- **NVIDIA driver:** a recent driver that supports CUDA 12.x (e.g. ≥ 525).
- **CUDA toolkit:** **12.4 – 12.6** (the build uses `nvcc` 12.x).
- **C/C++ compiler:** `gcc`/`g++` **≤ 13** (CUDA 12.6 supports gcc up to 13).
- **Conda** (Miniconda/Anaconda) for environment management.
- **Git** with submodule support.
- **Line endings:** every shell script (e.g. `generate.sh`, `wsl_generate.sh`, `install.sh`) must use **Unix-style LF** line endings — not CR or CRLF — because the pipeline runs on Ubuntu Linux, not Windows. A CRLF script fails with `./generate.sh: line N: $'\r': command not found`. If you edit a script on Windows, convert it before running: `sed -i 's/\r$//' <script>` (or `dos2unix <script>`). The repo enforces this via a `.gitattributes` (`*.sh text eol=lf`).

---

## 2. Prerequisites

### 2.1 Install Conda
If you don't have it, install [Miniconda](https://docs.conda.io/en/latest/miniconda.html):

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# restart the shell, then:
conda --version
```

### 2.2 Install the CUDA toolkit
The CUDA extensions (`simple-knn`, `pytorch3d`, `diff-gaussian-rasterization`, `ODGS`, `flash-attn`) are compiled against the CUDA toolkit at build time, so you need a **standalone CUDA 12.x toolkit** (not just the driver).

**Option A — package manager (Ubuntu, requires root):**
```bash
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-6
```

**Option B — NVIDIA runfile (no root needed):**
```bash
# Download the CUDA 12.6.2 runfile
wget https://developer.download.nvidia.com/compute/cuda/12.6.2/local_installers/cuda_12.6.2_560.35.03_linux.run
# Install only the toolkit to a user-writable path
sudo sh cuda_12.6.2_560.35.03_linux.run --toolkit --silent --no-drm --toolkitpath=/home/$USER/cuda-12.6
# (without root, use: sh cuda_12.6.2_560.35.03_linux.run --toolkit --silent --no-drm --toolkitpath=/home/$USER/cuda-12.6)
```

Verify:
```bash
/home/$USER/cuda-12.6/bin/nvcc --version   # should report release 12.6
```

> **Note:** If you install CUDA via conda (e.g. `cuda-nvcc`), make sure it is **12.x**, not 13.x — a 13.x toolkit in the env will shadow the system one and break the extension builds.

### 2.3 Verify the GPU
```bash
nvidia-smi
```
You should see your GPU and a driver that supports CUDA 12.x.

---

## 3. Clone the repository

```bash
git clone --recursive https://github.com/SkyworkAI/Matrix-3D.git
cd Matrix-3D
```

> `--recursive` is important: it pulls the nested submodules (`nvdiffrast`, `ODGS`, `simple-knn`, and the `glm` header library used by the rasterizers). If you cloned without it, run:
> ```bash
> git submodule update --init --recursive
> ```

---

## 4. Create the conda environment

```bash
conda create -n matrix3d python=3.10 -y
conda activate matrix3d
```

> **Python 3.10** is required. Do not use 3.11+ (several pinned dependencies and the CUDA extensions target 3.10).

---

## 5. Install PyTorch (CUDA build)

Install the CUDA-enabled PyTorch **first**, before the rest of the dependencies, so the CUDA extensions can link against it:

```bash
pip install torch==2.7.1 torchvision==0.22.1
```

> The README references `torch==2.7.0 / torchvision==0.22.0`; `2.7.1 / 0.22.1` are the matching, verified-working pair and are required by `xformers==0.0.31`.

Verify CUDA is visible to PyTorch:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expected: 2.7.1+cu12x True NVIDIA ...
```

---

## 6. Set the build environment for CUDA extensions

The CUDA extensions are compiled during `pip install`. They need the CUDA toolkit and a compatible compiler on the path. **Export these in the same shell before running `install.sh`:**

```bash
export CUDA_HOME=/home/$USER/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
unset CUDA_CFLAGS CUDA_LDFLAGS
```

> **Why this matters:**
> - `CC`/`CXX` must point to a **system gcc/g++ ≤ 13**. If your conda env has a newer `gcc_linux-64`/`gxx_linux-64` (e.g. gcc 15), it will shadow the system compiler and the CUDA builds will fail.
> - `CUDA_HOME`/`PATH`/`LD_LIBRARY_PATH` ensure `nvcc` 12.x and the CUDA 12 runtime are used, not a conda-provided 13.x toolkit.

---

## 7. Run the installation script

```bash
chmod +x install.sh
./install.sh
```

`install.sh` does the following (in order):
1. Builds the local CUDA submodules: `nvdiffrast`, `simple-knn`, `diff-gaussian-rasterization-w-pose`, and `ODGS` (`odgs-gaussian-rasterization`).
2. Installs `DiffSynth-Studio` in editable mode.
3. Installs the Python dependencies (diffusers, transformers, flash-attn, xformers, pytorch3d, taming-transformers, open-clip, etc.).

> **After `install.sh`, pin two packages** (the stock install can pull versions that break the pipeline):
> ```bash
> pip install "opencv-python==4.10.0.84" scikit-learn
> ```
> - **OpenCV must be 4.x** — the 5.x wheels dropped the OpenEXR codec, and the pipeline saves depth maps as `.exr`. (See [7.1](#71-known-build-fixes-if-a-step-fails).)
> - **scikit-learn** is required by the StableSR step (`ldm/models/diffusion/ddpm.py`).

> **Tip:** run it with logging so you can inspect failures:
> ```bash
> ./install.sh 2>&1 | tee install.log
> ```

### 7.1 Known build fixes (if a step fails)

The stock `install.sh` assumes a clean, well-matched toolchain. On a fresh machine you may hit the following. Each has a concrete fix:

| Symptom | Cause | Fix |
| :-- | :-- | :-- |
| `simple_knn.cu: ... FLT_MAX undefined` | missing `<cfloat>` include | add `#include <cfloat>` after `#include "simple_knn.h"` in `submodules/simple-knn/simple_knn.cu` |
| `rasterizer_impl.h: ... std::uintptr_t / uint32_t undefined` | missing `<cstdint>` include | add `#include <cstdint>` after `#include <vector>` in the `rasterizer_impl.h` of both `submodules/ODGS/submodules/odgs-gaussian-rasterization/cuda_rasterizer/` and `submodules/diff-gaussian-rasterization-w-pose/cuda_rasterizer/` |
| `fatal error: glm/glm.hpp: No such file or directory` | the `third_party/glm` git submodule is empty | `cd submodules/diff-gaussian-rasterization-w-pose && git submodule update --init --recursive` |
| `error: command 'nvcc' failed` / wrong CUDA version | conda CUDA 13.x or gcc 15 shadowing the toolchain | apply the exports in [Section 6](#6-set-the-build-environment-for-cuda-extensions); remove conda `gcc_linux-64`/`gxx_linux-64` and any `cuda-*` 13.x packages from the env |
| `ModuleNotFoundError: pkg_resources` | `setuptools ≥ 81` removed `pkg_resources` | `pip install "setuptools<81"` |
| `taming` installs but `import taming` fails | `taming` has no top-level `__init__.py`, so `find_packages()` yields an empty wheel | install from a local clone patched to use `find_namespace_packages()` |
| `flash-attn` build fails | build isolation hides `torch` | `pip install flash-attn==2.7.4.post1 --no-build-isolation` (already in `install.sh`) |
| `import open3d` → `libusb-1.0.so.0: cannot open shared object file` | the `libusb` shared library is missing (open3d is needed by Step 1) | `conda install -c conda-forge libusb -y` (no root) |
| `./generate.sh: line N: $'\r': command not found` | the script has Windows **CRLF** line endings (breaks `\` continuations) | `sed -i 's/\r$//' generate.sh` (or `dos2unix generate.sh`) |
| `cv2.imwrite(...exr)` → `could not find a writer for the specified extension` | **OpenCV 5.x** wheels dropped the OpenEXR codec (the `OPENCV_IO_ENABLE_OPENEXR` flag does nothing). The pipeline needs EXR for depth maps. | `pip install "opencv-python==4.10.0.84"` (4.x wheels bundle OpenEXR; the scripts already set `OPENCV_IO_ENABLE_OPENEXR=1`) |
| `ModuleNotFoundError: No module named 'sklearn'` (StableSR step) | scikit-learn not installed (needed by `ldm/models/diffusion/ddpm.py`) | `pip install scikit-learn` |

> **General rule for CUDA extensions:** always build with `--no-build-isolation` so the already-installed `torch` is visible to the compiler:
> ```bash
> pip install --no-build-isolation <package-or-path>
> ```

---

## 8. Verify the installation

A read-only smoke test is provided. It checks the conda env, CUDA toolkit, GPU, PyTorch CUDA, pinned versions, a ~58-module import test, checkpoints, and submodules:

```bash
bash verify_setup.sh
```

A fully working environment reports **`RESULT: PASS`** (all modules import, all checkpoints present). The smoke test checks **56 modules** and exits 0 only if every one imports.

Notes on modules the test deliberately handles:
- **`basicsr`** — the test puts `code/StableSR` on `sys.path` first, so it verifies the **local** `basicsr` the pipeline actually uses (the pip package is broken on modern `torchvision` and is not checked).
- **`open3d`** — pipeline-critical (Step 1 imports it via `pano_init → worldgen`). It needs `libusb-1.0.so.0`, usually absent on a minimal system. If it reports `FAIL`, fix it **without root**:
  ```bash
  conda install -c conda-forge libusb -y
  ```
- **`streamlit`** — intentionally **not installed** (it pins `protobuf<4`, conflicting with tensorboard's `protobuf>=6.31.1`); only the optional DiffSynth web app needs it, so it is not checked.
- **`realesrgan`** (pip) — broken on modern `torchvision`; the pipeline uses the local `code/StableSR` stack instead, so it is not checked.

You can also run the import smoke test directly:
```bash
python verify_env.py
```

---

## 9. Download the checkpoints

```bash
python code/download_checkpoints.py
```

This downloads 8 model files (~15–25 GB) from Hugging Face into `./checkpoints/`:

| File | Destination |
| :-- | :-- |
| `model.pt` (MoGe) | `checkpoints/moge/` |
| `stablesr_turbo.ckpt` | `checkpoints/StableSR/` |
| `vqgan_cfw_00011.ckpt` | `checkpoints/StableSR/` |
| `text2panoimage_lora.safetensors` | `checkpoints/flux_lora/` |
| `pano_lrm_480p.pt` | `checkpoints/pano_lrm/` |
| `pano_video_gen_480p.ckpt` | `checkpoints/Wan-AI/wan_lora/` |
| `pano_video_gen_720p.bin` | `checkpoints/Wan-AI/wan_lora/` |
| `pano_video_gen_720p_5b.safetensors` | `checkpoints/Wan-AI/wan_lora/` |

> **Optional (VideoSR only):** `venhancer_v2.pt` → `code/VideoSR/checkpoints/`. Only needed if you use the VideoSR enhancement step.
>
> **Behind a firewall / in China:** uncomment `os.environ["HF_ENDPOINT"] = 'https://hf-mirror.com'` at the top of `code/download_checkpoints.py`.

### 9.1 Hugging Face access (required)

Two models are **not** covered by `download_checkpoints.py` and are fetched from Hugging Face at runtime:

1. **`black-forest-labs/FLUX.1-dev`** (Step 1) — this is a **gated** repo. You must:
   - Accept the license on the [model page](https://huggingface.co/black-forest-labs/FLUX.1-dev), and
   - Authenticate, e.g. `huggingface-cli login` (or export `HF_TOKEN=hf_...`).
   Without this, Step 1 fails with `401 GatedRepoError`.
2. **Wan base video model** (Step 2) — downloaded at runtime into `checkpoints/Wan-AI/`:
   - `Wan-AI/Wan2.2-TI2V-5B` (~10–20 GB) when using `--use_5b_model`, or
   - `Wan-AI/Wan2.1-I2V-14B-720P` / `Wan2.1-I2V-14B-480P` (~28 GB each) otherwise.
   The first Step 2 run will therefore show a large download; subsequent runs reuse the cache.

---

## 10. Usage

### One-command generation
```bash
./generate.sh
```

> **On a 24 GB GPU (e.g. RTX 3090):** run `./generate.sh --low-vram` — this adds `--use_5b_model` to Step 2 so the 720p video step uses the light-weight 5B model (~12 GB) instead of the ~60 GB default. `--vram-mgmt` (adds `--enable_vram_management`, ~19 GB) is an alternative. Without one of these flags, Step 2 will OOM on a 24 GB card.
>
> **From Windows/WSL2:** run it inside the WSL `matrix3d` env, e.g. `wsl -- bash /path/to/Matrix-3D/wsl_generate.sh --low-vram` (a helper that activates the env, checks CUDA, then runs `./generate.sh` with the same arguments).

### Step by step

**Step 1 — Text/Image → Panorama image**
```bash
# text-to-panorama
python code/panoramic_image_generation.py \
    --mode=t2p \
    --prompt="a medieval village, half-timbered houses, cobblestone streets, lush greenery, clear blue sky, detailed textures, vibrant colors, high resolution" \
    --output_path="./output/example1"

# or image-to-panorama
python code/panoramic_image_generation.py \
    --mode=i2p \
    --input_image_path="./data/image1.jpg" \
    --output_path="./output/example1"
```

**Step 2 — Panorama image → Panoramic video**
```bash
VISIBLE_GPU_NUM=1
torchrun --nproc_per_node ${VISIBLE_GPU_NUM} code/panoramic_image_to_video.py \
  --inout_dir="./output/example1" \
  --resolution=720 \
  --use_5b_model   # recommended on <=24 GB GPUs (e.g. RTX 3090)
```
- `--resolution` is `480` or `720`.
- **5B model** (~12 GB, faster): add `--use_5b_model` — **recommended on 24 GB cards**; without it the 720p step needs ~60 GB and will OOM.
- **Low-VRAM mode** (~19 GB): add `--enable_vram_management` (alternative to the 5B model).
- Multi-GPU: raise `VISIBLE_GPU_NUM`.

**Step 3 — Panoramic video → 3D scene**
```bash
python code/panoramic_video_to_3DScene.py \
    --inout_dir="./output/example1" \
    --resolution=720
```
Two reconstruction options are available: a high-quality **optimization-based** method (~10 GB VRAM) and an efficient **feed-forward** method (PanoLRM, higher VRAM). See the README for details.

---

## 11. Troubleshooting

- **`torch.cuda.is_available()` is `False`** — the driver doesn't support the CUDA runtime, or `LD_LIBRARY_PATH` points at the wrong CUDA. Re-check [Section 6](#6-set-the-build-environment-for-cuda-extensions) and `nvidia-smi`.
- **A CUDA extension fails to build** — confirm `nvcc --version` reports 12.x and `gcc --version` reports ≤ 13, then rebuild that package with `pip install --no-build-isolation <pkg>`.
- **`ModuleNotFoundError: pkg_resources`** — `pip install "setuptools<81"`.
- **`glm/glm.hpp: No such file or directory`** — the `glm` submodule is empty; run `git submodule update --init --recursive` inside `submodules/diff-gaussian-rasterization-w-pose`.
- **`import open3d` → `libusb-1.0.so.0: cannot open shared object file`** — open3d (needed by Step 1) is missing the `libusb` library. Install it into the env without root: `conda install -c conda-forge libusb -y`.
- **`./generate.sh: line N: $'\r': command not found`** — the script has Windows CRLF line endings (common when edited on Windows), which break the `\` line continuations. Convert it: `sed -i 's/\r$//' generate.sh` (or `dos2unix generate.sh`).
- **Step 1 fails with `401 GatedRepoError` (FLUX.1-dev)** — the FLUX.1-dev repo is gated. Accept its license on the [Hugging Face page](https://huggingface.co/black-forest-labs/FLUX.1-dev), then authenticate with `huggingface-cli login` (or `export HF_TOKEN=hf_...`). See [Section 9.1](#91-hugging-face-access-required).
- **Step 1 fails with `RuntimeError: CUDA error: out of memory`** — FLUX.1-dev is ~24 GB (transformer) + ~12 GB (text encoders) in bf16 and won't fit on a 24 GB GPU if loaded whole. The code uses `enable_sequential_cpu_offload()` to stream layers one at a time (peak VRAM a few GB). If you see this OOM, confirm `code/panoramic_image_generation.py` is calling `enable_sequential_cpu_offload()` (not `.to(device)` / `enable_model_cpu_offload()`).
- **`AttributeError: module 'utils3d.numpy' has no attribute 'icosahedron'`** (or `image_uv`, `sliding_window_`, …) — a **pip-installed `utils3d` (v1.7)** in site-packages is shadowing the repo-local `code/MoGe/utils3d`, which has the functions MoGe needs. The pipeline scripts now `sys.path.insert(0, ...)` the local dirs so they win. If you revert those, or add a new script that imports `moge`/`utils3d`, make sure the local `code/MoGe` dir is at the **front** of `sys.path` (not appended).
- **`ModuleNotFoundError: No module named 'torchvision.transforms.functional_tensor'`** (in the StableSR step) — a **pip-installed `basicsr` (v1.4.2)** is shadowing the repo-local `code/StableSR/basicsr`. The pip version imports `torchvision.transforms.functional_tensor`, which was removed in torchvision 0.22; the local `basicsr` uses `torchvision.transforms._functional_tensor` (with underscore), which works. `code/StableSR/scripts/sr_val_ddpm_text_T_vqganfin_old.py` now inserts the local `code/StableSR` dir at the front of `sys.path`.
- **`cv2.imwrite(...)` → `could not find a writer for the specified extension`** (or `OpenEXR codec is disabled`) — the depth maps are saved as **EXR**, which requires an OpenCV build with the OpenEXR codec. **OpenCV 5.x wheels do not include it**; use a 4.x wheel: `pip install "opencv-python==4.10.0.84"`. The pipeline scripts already set `OPENCV_IO_ENABLE_OPENEXR=1`, so once the codec is present, EXR read/write works.
- **`ModuleNotFoundError: No module named 'sklearn'`** (StableSR step) — `pip install scikit-learn`.
- **Step 2 hangs / downloads a huge file on first run** — expected: the Wan base video model (`Wan2.2-TI2V-5B` or `Wan2.1-I2V-14B-*`) is fetched at runtime, not by `download_checkpoints.py`. See [Section 9.1](#91-hugging-face-access-required).
- **WSL2 notes:** `/tmp` is cleared on WSL restart (re-download the CUDA runfile if needed). `sudo` may require a password — prefer the no-root runfile install in [Section 2.2](#22-install-the-cuda-toolkit). Build CUDA extensions from a **copy** of the repo (e.g. in `/tmp`) to avoid the `from . import _C` circular import that in-place builds can trigger.
- **Disk cleanup:** after a successful install you can remove the CUDA runfile (`~4 GB`) and run `conda clean -a` (`~8 GB`).

---

## 12. Quick reference

```bash
# 1. Prereqs: conda + CUDA 12.x toolkit + gcc<=13 (Sections 2-3)
git clone --recursive https://github.com/SkyworkAI/Matrix-3D.git && cd Matrix-3D

# 2. Environment
conda create -n matrix3d python=3.10 -y && conda activate matrix3d
pip install torch==2.7.1 torchvision==0.22.1

# 3. Build env (Section 6)
export CUDA_HOME=/home/$USER/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CC=/usr/bin/gcc CXX=/usr/bin/g++
unset CUDA_CFLAGS CUDA_LDFLAGS

# 4. Install
./install.sh

# 5. Verify
bash verify_setup.sh

# 6. Checkpoints
python code/download_checkpoints.py

# 7. Generate
./generate.sh
# (from Windows/WSL2: wsl -- bash /path/to/Matrix-3D/wsl_generate.sh)
```
