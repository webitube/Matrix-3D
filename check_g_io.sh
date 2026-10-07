#!/usr/bin/env bash
# check_g_io.sh — diagnose what still reads/writes the G drive during the pipeline.
# Run from WSL:  bash /mnt/d/Dev/Matrix-3D/check_g_io.sh

echo "=== 1. Live processes with cwd or open files on /mnt/g ==="
for p in /proc/[0-9]*/cwd; do
    t=$(readlink "$p" 2>/dev/null)
    case "$t" in /mnt/g*) echo "  PID $(basename "$(dirname "$p")"): cwd=$t";; esac
done
for p in /proc/[0-9]*/fd; do
    d=$(dirname "$p")
    pid=$(basename "$d")
    for f in "$p"/*; do
        t=$(readlink "$f" 2>/dev/null)
        case "$t" in /mnt/g*) echo "  PID $pid: open $t";; esac
    done
done 2>/dev/null

echo
echo "=== 2. Symlinks anywhere in the D project pointing at /mnt/g ==="
find /mnt/d/Dev/Matrix-3D -type l -lname '/mnt/g*' 2>/dev/null

echo
echo "=== 3. Symlinks in the conda env pointing at /mnt/g ==="
find /home/ron/miniconda3/envs/matrix3d -type l -lname '/mnt/g*' 2>/dev/null | head -20

echo
echo "=== 4. HF / torch / pip cache locations (matrix3d env) ==="
source /home/ron/miniconda3/etc/profile.d/conda.sh
conda activate matrix3d
python - <<'EOF'
import os
print("  HF_HUB_CACHE :", os.path.expanduser("~/.cache/huggingface/hub"))
print("  TORCH_HOME   :", os.environ.get("TORCH_HOME", "(unset -> ~/.cache/torch)"))
print("  XDG_CACHE    :", os.environ.get("XDG_CACHE_HOME", "(unset -> ~/.cache)"))
try:
    import huggingface_hub
    from huggingface_hub import constants
    print("  HF_HUB_CACHE (lib):", constants.HF_HUB_CACHE)
except Exception as e:
    print("  HF_HUB_CACHE (lib): n/a", e)
EOF

echo
echo "=== 5. Anything in the G copy that is NOT in the D copy (stale data) ==="
du -sh /mnt/g/Dev/Matrix-3D 2>/dev/null
echo "  (checkpoints/ in the G copy is a symlink to D — that part is fine)"

echo
echo "=== 6. Recent logs in the G copy ==="
find /mnt/g/Dev/Matrix-3D -name '*.log' -newermt '2026-10-01' 2>/dev/null | head

echo
echo "=== 7. Mount options for /mnt/g and /mnt/d ==="
grep -E 'mnt/(g|d)' /proc/mounts

echo
echo "=== 8. WSL metadata (metadata=csproj / case-sensitive etc.) ==="
cat /etc/wsl.conf 2>/dev/null
