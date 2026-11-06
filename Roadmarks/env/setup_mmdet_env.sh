#!/usr/bin/env bash
# Conda env "mmdet" for MMDetection 3.3: torch 2.7.1 + CUDA 12.8 (works on RTX 50xx / sm_120), mmcv 2.1.0 built
# from source. Clones mmcv and mmdetection into the workspace (the folder that holds this repo).
#   bash Roadmarks/env/setup_mmdet_env.sh            (set ARCH for another GPU, e.g. ARCH=8.6 for RTX 30xx)
set -exo pipefail   # no -u: conda activate hooks use unset variables
WS=$(cd "$(dirname "$0")/../.." && pwd)
ARCH=${ARCH:-12.0}
source "$(conda info --base)/etc/profile.d/conda.sh"
conda env list | grep -q "^mmdet " || conda create -n mmdet python=3.10 -y
conda activate mmdet
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
conda install -c nvidia/label/cuda-12.8.1 cuda-nvcc cuda-cudart-dev cuda-libraries-dev -y   # nvcc 12.8 inside the env
pip install "setuptools<70" wheel ninja psutil pycocotools mmengine==0.10.7 opencv-python shapely
export CUDA_HOME=$CONDA_PREFIX

[ -d "$WS/mmcv" ] || git clone -b v2.1.0 --depth 1 https://github.com/open-mmlab/mmcv.git "$WS/mmcv"
cd "$WS/mmcv"
# AMP fix: NMS writes float32 scores into a float16 tensor
sed -i 's/scores_after_nms\[mask\[keep\]\] = dets\[:, -1\]$/scores_after_nms[mask[keep]] = dets[:, -1].to(scores_after_nms.dtype)  # AMP: half vs float/' mmcv/ops/nms.py
TORCH_CUDA_ARCH_LIST="$ARCH" MMCV_WITH_OPS=1 FORCE_CUDA=1 MAX_JOBS=16 python setup.py develop   # setuptools<70: no PEP 660

[ -d "$WS/mmdetection" ] || git clone -b v3.3.0 --depth 1 https://github.com/open-mmlab/mmdetection.git "$WS/mmdetection"
cd "$WS/mmdetection"
pip install -r requirements/runtime.txt
python setup.py develop --no-deps
python -c "import torch, mmcv, mmdet, mmengine; from mmcv.ops import RoIAlign; print('OK torch', torch.__version__, 'mmcv', mmcv.__version__, 'mmdet', mmdet.__version__, 'cuda', torch.cuda.is_available())"
