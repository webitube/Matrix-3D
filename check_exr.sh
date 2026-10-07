#!/usr/bin/env bash
source /home/ron/miniconda3/etc/profile.d/conda.sh
conda activate matrix3d
cd /mnt/g/Dev/Matrix-3D

echo "=== 1. cv2 version + EXR write/read round-trip ==="
python -c "
import cv2, numpy as np
print('cv2 version:', cv2.__version__)
a = np.random.rand(8, 8).astype(np.float32)
ok = cv2.imwrite('/tmp/test_exr.exr', a, [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])
print('EXR write:', ok)
b = cv2.imread('/tmp/test_exr.exr', cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
print('EXR read :', None if b is None else (b.shape, b.dtype))
"

echo
echo "=== 2. sklearn import ==="
python -c "import sklearn; from sklearn.decomposition import PCA; print('sklearn', sklearn.__version__, 'OK')"

echo
echo "=== 3. StableSR ddpm import chain (the exact failing import) ==="
cd code/StableSR
python -c "
import sys, os
sys.path.insert(0, os.path.abspath('.'))
from ldm.models.diffusion.ddpm import DDIMSampler
print('ldm.models.diffusion.ddpm import OK')
"
cd /mnt/g/Dev/Matrix-3D
