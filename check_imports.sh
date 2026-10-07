#!/usr/bin/env bash
source /home/ron/miniconda3/etc/profile.d/conda.sh
conda activate matrix3d
cd /mnt/g/Dev/Matrix-3D

echo "=== TEST 1: MoGe utils3d (should resolve to local, have icosahedron) ==="
cd code/MoGe
python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path('scripts/infer_panorama.py').absolute().parents[1]))
import utils3d
print('utils3d file:', utils3d.__file__)
print('has icosahedron:', hasattr(utils3d.numpy, 'icosahedron'))
v,_ = utils3d.numpy.icosahedron()
print('icosahedron vertices:', v.shape)
"
cd /mnt/g/Dev/Matrix-3D

echo
echo "=== TEST 2: StableSR basicsr (should resolve to local, import cleanly) ==="
cd code/StableSR
python -c "
import sys, os
_SR_ROOT = os.path.abspath('.')
sys.path.insert(0, _SR_ROOT)
import basicsr
print('basicsr file:', basicsr.__file__)
from basicsr.archs.arch_util import default_init_weights, make_layer, pixel_unshuffle
print('arch_util imports OK')
"
cd /mnt/g/Dev/Matrix-3D
