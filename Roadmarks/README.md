# Roadmarks: road-marking detection on the front camera

This repo trains and runs an instance-segmentation model that finds road markings in the undistorted front-camera
image. Its training labels are projected automatically from the NGII HD map; nothing is labelled by hand.

| Item | Value |
|---|---|
| Model | **RTMDet-Ins-s**, 10.2 M parameters, MMDetection 3.3 (Apache 2.0) |
| Classes (11) | `ahead`, `left`, `right`, `U-turn`, `ahead + right`, `ahead + left`, `left + U-turn`, `crosswalk`, `bicycle crossing`, `stop line`, `ahead + U-turn` |
| Input | Undistorted front image: fisheye, alpha = 0, 1920×1536 |
| Output | Class, score, box and mask polygon for each marking |
| Released checkpoint | `rtmdet-ins_s_roadmark_v1.pth`, about 41 MB, stored on the data server (see [Checkpoint](#checkpoint)) |

Most users only need to **run the model on images**: see [Inference only](#1-inference-only).
To rebuild the dataset, retrain, and redo all evaluations: see [Reproduce everything](#2-reproduce-everything).

---

## Repository

| File | What it does |
|---|---|
| `predict.py` | Runs the model on any folder of images or a video. Writes JSON, plus annotated images or a video if asked. |
| `hdmap_gt.py` | Projects the HD map (B3 surface marks, B2 stop lines) onto every front frame of a session. |
| `build_roadmark_dataset.py` | Builds the dataset: per-frame ground truth, occlusion from anno-3DOD boxes, keeps only frames that show a marking, undistorts them. |
| `to_coco.py` | Converts the dataset to COCO instance segmentation (`train.json` / `val.json`). Hidden, far or rare markings become ignore regions. |
| `configs/rtmdet-ins_s_roadmark.py` | Training config: 20 epochs, no flip, full-resolution testing. |
| `rtmdet_ignore.py` | Small RTMDet patch: background points inside ignore regions get no classification loss. |
| `vis_val.py` | Side-by-side images on the validation set: HD-map label next to prediction. |
| `infer_sessions.py` | Runs the model on every frame of whole sessions. Writes a side-by-side video, prediction JSON and metrics per session. |
| `event_eval.py` | Marking-level score from the `infer_sessions.py` output: found %, wrong type, false alarms. |
| `export_checkpoint.py` | Turns a training checkpoint into a small release checkpoint (weights and class names only). |
| `run_undistort.py` | Optional. Undistorts whole sessions (all 6 cameras) with MotifE2E_Demo. The dataset build does not need it. |
| `env/setup_mmdet_env.sh` | Creates the conda env `mmdet`: torch 2.7.1 + CUDA 12.8, mmcv 2.1.0 built from source, MMDetection 3.3.0. |

### Workspace layout

All scripts expect this layout. `WS` is any folder on your **local** machine:

```
$WS/
├── Roadmarks/              this repo
├── mmdetection/            cloned by env/setup_mmdet_env.sh (v3.3.0)
├── mmcv/                   cloned and built by env/setup_mmdet_env.sh (v2.1.0)
├── MotifE2E_Demo/          only for building the dataset or for raw fisheye input (calibration / undistortion code)
├── prep/                   vehicle calibration *__camera_config.json, *__lidar_config.json ... (copied from the server)
├── checkpoints/            released checkpoint (copied from the server)
├── roadmark_dataset_202/   built dataset (about 97 GB)
└── work_dirs/              training runs and evaluation outputs
```

---

## 0. Setup (everybody)

```bash
export WS=~/roadmarks_ws                      # local workspace, any path
mkdir -p $WS && cd $WS
git clone <this repo url> Roadmarks

# conda env "mmdet". Takes about 15 min because mmcv is compiled.
# The default is an RTX 50xx GPU (sm_120); for other GPUs set ARCH, e.g. ARCH=8.6 (RTX 30xx) or ARCH=8.9 (RTX 40xx).
bash Roadmarks/env/setup_mmdet_env.sh

conda activate mmdet
export PYTHONPATH=$WS/Roadmarks               # the configs import rtmdet_ignore.py
```

Mount the data server read-only with sshfs. **Never write into the server folders**; every script writes locally.

```bash
export SERVER=~/server-1.201.66.58
mkdir -p $SERVER
sshfs -o ro,reconnect <user>@1.201.66.58:/mnt/ssd2/inhouse/Vb12ddde/ $SERVER
ls $SERVER        # prep/  sessions/  20261029_185000_e2e_prepare_202sessions/ ...
```

### Checkpoint

The released checkpoint is kept on the server and not in git (`*.pth` is git-ignored). Copy the latest one into the workspace:

```bash
mkdir -p $WS/checkpoints
cp $SERVER/<models folder>/rtmdet-ins_s_roadmark_v1.pth $WS/checkpoints/
```

---

## 1. Inference only

You need only the setup above and the checkpoint. The dataset, HD map and MotifE2E_Demo are not needed for
undistorted input.

```bash
cd $WS
# a folder of UNDISTORTED front images (1920x1536)
python Roadmarks/predict.py --ckpt checkpoints/rtmdet-ins_s_roadmark_v1.pth \
       --input /path/to/undistorted_front_images --out out/my_run --vis

# RAW fisheye front images straight from a session: undistorted on the fly
# (needs MotifE2E_Demo in $WS and the vehicle calibration in prep/)
cp -r $SERVER/prep $WS/prep
python Roadmarks/predict.py --ckpt checkpoints/rtmdet-ins_s_roadmark_v1.pth \
       --input $SERVER/sessions/20260910_070626/perception_sensor_data/cam-rgb-front \
       --raw --prep prep --out out/20260910_070626 --video

# a video file works too
python Roadmarks/predict.py --ckpt checkpoints/rtmdet-ins_s_roadmark_v1.pth --input drive.mp4 --out out/drive --video
```

| Option | Meaning |
|---|---|
| `--score 0.3` | Score threshold. RTMDet scores run lower than Mask R-CNN's; 0.3 is a good default. |
| `--vis` | Writes annotated images to `<out>/vis/`. |
| `--video` | Writes `<out>/predictions.mp4` (frames in file-name order, `--fps 10`). |
| `--device cpu` | Runs without a GPU (about 5 s per image instead of a few tens of ms). |

`<out>/predictions.json`:
```json
{"classes": [...], "score_thr": 0.3,
 "images": [{"image": "xxx.jpg", "width": 1920, "height": 1536,
             "predictions": [{"class": "ahead", "score": 0.91, "bbox_xyxy": [x0, y0, x1, y1],
                              "mask_polygons": [[x, y, x, y, ...]]}]}]}
```
All coordinates are pixels of the **undistorted** front image.

From your own Python code:
```python
import sys; sys.path.insert(0, 'Roadmarks'); import rtmdet_ignore   # registers the custom head
from mmdet.apis import init_detector, inference_detector
model = init_detector('Roadmarks/configs/rtmdet-ins_s_roadmark.py', 'checkpoints/rtmdet-ins_s_roadmark_v1.pth', device='cuda:0')
r = inference_detector(model, bgr_image).pred_instances      # .labels .scores .bboxes .masks
```

---

## 2. Reproduce everything

All commands run from `$WS` with `conda activate mmdet` and `PYTHONPATH=$WS/Roadmarks` set.
Data is read from the mounted server and everything is written locally.

### 2.1 Local inputs

```bash
cd $WS
git clone https://github.com/motifdrive/MotifE2E_Demo.git     # calibration + undistortion code (internal)
cp -r $SERVER/prep $WS/prep                                    # vehicle calibration
```

### 2.2 Build the dataset

The dataset uses the 202 sessions of the e2e export, keeping only frames with an arrow, crosswalk or stop line.
Building it takes several hours, mostly spent reading images over sshfs.

```bash
python Roadmarks/build_roadmark_dataset.py \
       --prepare $SERVER/20261029_185000_e2e_prepare_202sessions \
       --sessions-root $SERVER/sessions --prep $WS/prep \
       --out $WS/roadmark_dataset_202 --workers 12
# test first with --limit 2; re-running skips finished sessions; --only gt | build runs one stage
```

The build has two stages:
- **gt:** `hdmap_gt.py` projects the HD map into every front frame and writes `gt/<session>.json`.
- **build:** checks occlusion against the anno-3DOD boxes, keeps a frame if it has at least one marking that is
  visible ≥ 50 % and ≤ 30 m away (crossings and stop lines also when ≥ 4000 px² or ≥ 200 px long are visible),
  undistorts those frames to `images/<session>/` and writes `labels/<session>.json`.

Result: 133,149 of 232,457 frames kept.

### 2.3 Convert to COCO

```bash
python Roadmarks/to_coco.py --data $WS/roadmark_dataset_202
```

The split has no session leakage: day **20260910 is validation**, except two sessions that move to training
because they hold most of the rare `ahead + U-turn` examples. Training uses every 2nd frame and validation every
5th.

| Split | Sessions | Images | Instances | Ignore regions |
|---|---|---|---|---|
| Train | 171 | 56,137 | 164,419 | 127,854 |
| Val | 31 | 4,225 | 13,069 | 9,206 |

### 2.4 Train

```bash
mkdir -p mmdetection/checkpoints
wget -P mmdetection/checkpoints \
  https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet-ins_s_8xb32-300e_coco/rtmdet-ins_s_8xb32-300e_coco_20221121_212604-fdc5d7ec.pth

python mmdetection/tools/train.py Roadmarks/configs/rtmdet-ins_s_roadmark.py
# -> work_dirs/rtmdet-ins_s_roadmark/  (best_coco_segm_mAP_epoch_N.pth, logs)
# resume after an interruption: add --resume
```

Training settings:
- Starts from COCO weights.
- 20 epochs, batch 16, AdamW with lr 0.001, cosine decay over epochs 10–20, mixed precision (AMP), EMA weights.
- Augmentation: random scale 0.5–1.5×, random 1280×1024 crop, colour jitter. **No flip** (it would turn a left
  arrow into a right arrow), and no Mosaic or MixUp.
- Validation every 2 epochs, then every epoch for the last 4.
- On one RTX 5090 (32 GB): about 6.5 h, 16 GB of GPU memory.

### 2.5 Evaluate

The released checkpoint loads directly. Raw training checkpoints need
`TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`, because torch ≥ 2.6 refuses mmengine objects by default.

```bash
CKPT=checkpoints/rtmdet-ins_s_roadmark_v1.pth          # or work_dirs/rtmdet-ins_s_roadmark/best_coco_segm_mAP_epoch_N.pth

# COCO mAP on the validation set (box + mask, per class)
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python mmdetection/tools/test.py Roadmarks/configs/rtmdet-ins_s_roadmark.py $CKPT

# 40 side-by-side validation images (HD-map label | prediction) -> <ckpt dir>/val_vis/<ckpt name>/
python Roadmarks/vis_val.py --ckpt $CKPT --n 40 --score 0.3

# every frame of whole sessions: video + predictions json + metrics per session, summary.csv
python Roadmarks/infer_sessions.py --ckpt $CKPT --sessions '20260910_*' \
       --sessions-root $SERVER/sessions --prep $WS/prep --out work_dirs/infer_20260910 --score 0.3

# marking-level score (found %, false alarms) for the same sessions -> event_summary.json
python Roadmarks/event_eval.py --dir work_dirs/infer_20260910 --sessions-root $SERVER/sessions --score 0.3
```

The scores in `event_eval.py` mean:
- **Found:** a marking (one HD-map id) gets the right class, with mask IoU ≥ 0.3 and 16 px tolerance, in at least 1 frame.
- **False alarm:** a detection lasting at least 3 frames where the HD map has no marking.

The overall figure skips sessions that were used in training. `infer_sessions.py` needs the `gt/` folder from
step 2.2; run `build_roadmark_dataset.py --only gt` for sessions that are not in the dataset.

### 2.6 Release a new checkpoint

```bash
python Roadmarks/export_checkpoint.py work_dirs/rtmdet-ins_s_roadmark/best_coco_segm_mAP_epoch_N.pth \
       checkpoints/rtmdet-ins_s_roadmark_v2.pth     # weights + class names only, about 41 MB
# then copy it to the models folder on the server and note the version here
```

