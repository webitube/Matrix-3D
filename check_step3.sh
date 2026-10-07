#!/usr/bin/env bash
source /home/ron/miniconda3/etc/profile.d/conda.sh
conda activate matrix3d
cd /mnt/g/Dev/Matrix-3D

echo "=== Syntax check all edited files ==="
for f in \
  code/MoGe/scripts/infer_panorama.py \
  code/StableSR/scripts/sr_val_ddpm_text_T_vqganfin_old.py \
  code/utils_3dscene/panorama_video_to_perspective_depth_sequential.py \
  code/utils_3dscene/gs_optim_datagen.py \
  code/utils_3dscene/pipeline_utils_3dscene.py
do
  if python -m py_compile "$f"; then echo "OK   $f"; else echo "FAIL $f"; fi
done

echo
echo "=== utils3d resolution from each Step 3 script's CWD (project root) ==="
python -c "
import os, sys
# Simulate gs_optim_datagen.py header
_CODE_DIR = os.path.abspath('code')
sys.path.insert(0, os.path.join(_CODE_DIR, 'MoGe'))
sys.path.insert(0, _CODE_DIR)
import utils3d
print('utils3d file :', utils3d.__file__)
print('has image_uv :', hasattr(utils3d.numpy, 'image_uv'))
print('has icosahedron:', hasattr(utils3d.numpy, 'icosahedron'))
import moge
print('moge file    :', moge.__file__)
"
