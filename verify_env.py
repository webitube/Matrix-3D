"""Smoke test: verify all Matrix-3D dependencies import correctly."""
import importlib
import os
import sys

print(f"Python: {sys.version.split()[0]}")

# The pipeline uses the LOCAL basicsr (code/StableSR/basicsr), not the pip
# package (which is broken on modern torchvision). Put it on the path so the
# `basicsr` check below tests the one that actually matters.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "code", "StableSR"))

# xfuser logs a long DEBUG traceback for an optional Flux2 pipeline that this
# diffusers version doesn't provide. It's harmless; quiet it for clean output.
# xfuser reads LOG_LEVEL at import time (default "debug"), so set it before
# the import loop below.
os.environ.setdefault("LOG_LEVEL", "warning")

mods = [
    # Core
    "torch", "torchvision", "xformers", "transformers", "flash_attn",
    # CUDA extensions
    "nvdiffrast", "pytorch3d", "utils3d", "simple_knn",
    "diff_gaussian_rasterization", "odgs_gaussian_rasterization",
    # Pipeline
    "diffsynth", "taming", "clip", "open_clip",
    # Supporting
    # (streamlit intentionally not installed: it pins protobuf<4, which
    #  conflicts with tensorboard's protobuf>=6.31.1; only the optional
    #  DiffSynth web app needs it.)
    # (realesrgan pip package is broken on modern torchvision; the pipeline
    #  uses the local code/StableSR stack instead.)
    "tensorboard", "basicsr",
    "decord", "einops", "ftfy", "regex", "sentencepiece", "timm",
    "lpips", "trimesh", "kornia", "onnx",
    "av", "imageio", "huggingface_hub", "accelerate", "safetensors",
    "cv2", "scipy", "numpy", "PIL", "tqdm", "skimage",
    "pytorch_lightning", "omegaconf", "diffusers", "modelscope",
    "peft", "easydict", "torchsde", "fairscale", "natsort",
    "jaxtyping", "matplotlib", "torchmetrics", "webdataset",
    "plyfile", "pyrender", "open3d", "py360convert", "xfuser",
]

ok, fail = 0, 0
for m in mods:
    try:
        mod = importlib.import_module(m)
        v = getattr(mod, "__version__", "?")
        print(f"OK    {m:32s} {v}")
        ok += 1
    except Exception as e:
        print(f"FAIL  {m:32s} {type(e).__name__}: {str(e)[:90]}")
        fail += 1

print()
import torch
print(f"torch.cuda.is_available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"torch.backends.cudnn.version: {torch.backends.cudnn.version()}")

print(f"\n{ok} OK, {fail} FAILED")
sys.exit(1 if fail else 0)
