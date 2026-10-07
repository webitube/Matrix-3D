#!/usr/bin/env bash
# Test: does the LOCAL code/StableSR/basicsr import cleanly (pipeline context)?
source /home/ron/miniconda3/etc/profile.d/conda.sh
conda activate matrix3d
export CUDA_HOME=/home/ron/cuda-12.6
export PATH=/home/ron/cuda-12.6/bin:$PATH
export LD_LIBRARY_PATH=/home/ron/cuda-12.6/lib64:$LD_LIBRARY_PATH

echo "=== local basicsr import (from code/StableSR, like the pipeline) ==="
cd /mnt/g/Dev/Matrix-3D/code/StableSR
python -c "
import sys
sys.path.insert(0, '.')
import basicsr
print('basicsr file:', basicsr.__file__)
from basicsr.utils import get_root_logger
print('local basicsr OK')
" 2>&1 | tail -8

echo
echo "=== local ldm import (StableSR core) ==="
cd /mnt/g/Dev/Matrix-3D/code/StableSR
python -c "
import sys
sys.path.insert(0, '.')
from ldm.util import instantiate_from_config
print('local ldm OK')
" 2>&1 | tail -8

echo
echo "=== pytorch_lightning import (current protobuf) ==="
python -c "import pytorch_lightning; print('pytorch_lightning OK', pytorch_lightning.__version__)" 2>&1 | tail -4

echo
echo "=== protobuf version ==="
python -c "import google.protobuf; print('protobuf', google.protobuf.__version__)" 2>&1 | tail -2
