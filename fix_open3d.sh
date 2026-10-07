#!/usr/bin/env bash
# Install libusb into the conda env (no root) so open3d can import.
source /home/ron/miniconda3/etc/profile.d/conda.sh
conda activate matrix3d

echo "=== conda install libusb (conda-forge) ==="
conda install -c conda-forge libusb -y 2>&1 | tail -20

echo
echo "=== locate libusb-1.0.so.0 in env ==="
find /home/ron/miniconda3/envs/matrix3d -name "libusb-1.0.so*" 2>/dev/null

echo
echo "=== open3d import test ==="
python -c "import open3d as o3d; print('open3d OK, version', o3d.__version__)" 2>&1 | tail -5
