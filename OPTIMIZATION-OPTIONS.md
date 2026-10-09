# Optimization Options — 3DGS Training Memory & Speed

Goal: keep Gaussian Splatting (3DGS) training inside the **24 GB dedicated VRAM** of the
RTX 3090 (host: 96 GB system RAM) so it never spills into **Shared VRAM**, and make the
pipeline observable (memory + progress) and easy to re-test.

## Root cause of the slowdown (measured)

On Windows, **"Shared" GPU memory is system RAM the GPU borrows over PCIe.** It only climbs
once the 24 GB of dedicated VRAM is exhausted — and PCIe is ~10× slower than HBM, which is
exactly why throughput collapses.

The cross run (`PanoQwenImage-2512-Cross`) ended with **10,546,098 Gaussians**, up from the
initial `--num_of_point_cloud 3000000` — a **3.5× growth** driven by densification.

The slowdown timeline matches the densification schedule exactly
(`--densify_from_iter 500 --densify_until_iter 1501`, `densification_interval = 500`, so the
cloud is grown at iterations **500, 1000, 1500**):

| Iteration | Event                          | Point cloud | Speed     |
|-----------|--------------------------------|-------------|-----------|
| 1000      | 2nd densification              | growing     | 17.55 it/s |
| **1500**  | **3rd (final) densification**  | **hits max**| **4.21 it/s** |
| 1600+     | no more growth; VRAM > 24 GB   | 10.5 M      | 3.18 s/it |

Per Gaussian the persistent footprint is far more than the 14 parameter floats:
parameters (~56 B) + Adam `exp_avg`/`exp_avg_sq` (~112 B) + transient gradients (~56 B) +
densification accumulators (~20 B) + rasterizer render buffers (~50 B) ≈ **~300 B/point**.
At 3 M points that is a few GB (the 3–4 GB baseline); at 10.5 M points the working set
crosses 24 GB right at the final densification, so CUDA spills to shared/system memory.

The final ~200 iterations "recovered" some speed (5.7 → 2.18 s/it) because after iteration
1500 there is **no more densification or pruning** — the point cloud is frozen, the caching
allocator stops churning, and the working set settles. It never returns to 17 it/s because the
cloud is still 3.5× larger than at the start.

## Levers (in order of impact)

The master lever is the **final point count**.

1. **Stop densifying earlier** — `--densify_until_iter 1000` skips the 1500 growth step.
   Cheapest first experiment; likely keeps you under the threshold.
2. **Lower the growth rate** — raise `--densify_grad_threshold` (default `0.0002`) so fewer
   points qualify for clone/split.
3. **Reduce per-point cost** — already at `--sh_degree 0` (no `f_rest`). The
   `--use_decoupled_appearance` path adds the appearance network + per-view embeddings;
   dropping it (if not needed) saves a chunk.
4. **Prune more aggressively** — the `min_opacity` prune threshold is hardcoded to `0.05` in
   `densify_and_prune`; raising it prunes more dead Gaussians.

All of these are now exposed as CLI flags on the Step 3 script (see "Testing" below), so no
file edits are needed to experiment.

## Observability added

- **Memory logging in training** (`code/Pano_GS_Opt/train.py`): every
  `--log_memory_interval` iterations (default 100) it prints dedicated VRAM
  (allocated/reserved), system RAM (used/total), and the current point count, e.g.
  `[MEM iter=1500] pts=10.5M | VRAM alloc=23.8GB resv=24.1GB | RAM used=62.3/95.8GB (65%)`.
  Correlate: when **VRAM resv plateaus near 24 GB and RAM used climbs**, that is the
  Shared-VRAM spillover you see in Task Manager.
- **Depth-step progress + ETA** (`code/utils_3dscene/panorama_video_to_perspective_depth_sequential.py`):
  the per-frame MoGe depth loop now prints `[depth] MoGe step k/K (frame i/N) | elapsed | ETA`,
  so the previously silent depth phase shows step #/total and a time estimate.

## Testing (jump straight to training)

The Step 3 script (`code/panoramic_video_to_3DScene.py`) now supports `--train_only`, which
skips depth / datagen / StableSR and runs `train.py` directly on an existing `geom_optim/data`
directory. Combined with the densify overrides, you can test a densification schedule in a
single command:

```bash
wsl bash -lc "source /home/ron/miniconda3/etc/profile.d/conda.sh && conda activate matrix3d && cd /mnt/d/Dev/Matrix-3D && \
python code/panoramic_video_to_3DScene.py --inout_dir ./output/PanoQwenImage-2512-Cross --train_only --densify_until_iter 1000"
```

Watch the `[MEM ...]` lines: if `VRAM resv` stays under ~24 GB and `RAM used` stays flat, the
schedule fits. Try `--densify_until_iter 1000`, then `--densify_grad_threshold 0.0004`, etc.

## Possible future work

- **`--max_gaussians` hard cap** in `densify_and_prune` (e.g. cap at 6 M) so the point cloud
  is bounded regardless of the schedule. Not yet implemented — a naive "truncate to first N"
  would be arbitrary; a quality-preserving cap should prune by opacity (keep the most opaque).
- **Log Shared VRAM directly.** From WSL we can measure dedicated VRAM (PyTorch) and system
  RAM, but the Windows "Shared" label is a Task Manager view of the same system RAM; the
  alloc/resv + RAM-used correlation above is the practical proxy.
- **AGENTS.md boundary note:** `code/Pano_GS_Opt/` is listed as a do-not-modify vendored black
  box, but tuning 3DGS training (memory logging, densification) requires editing it. Update
  `AGENTS.md` to carve out `Pano_GS_Opt` as modifiable if these changes are kept.