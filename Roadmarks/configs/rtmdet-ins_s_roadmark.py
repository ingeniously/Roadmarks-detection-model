# RTMDet-Ins-s (10.2 M params), COCO-pretrained, fine-tuned on the HD-map road-marking dataset (roadmark_dataset_202).
# Paths are relative to the workspace = the folder that holds this repo, mmdetection/, roadmark_dataset_202/ ...
# train (from the workspace):  PYTHONPATH=Roadmarks python mmdetection/tools/train.py Roadmarks/configs/rtmdet-ins_s_roadmark.py
_base_ = '../../mmdetection/configs/rtmdet/rtmdet-ins_s_8xb32-300e_coco.py'
WS = '{{fileDirname}}/../..'  # workspace
custom_imports = dict(imports=['rtmdet_ignore'], allow_failed_imports=False)  # Roadmarks/rtmdet_ignore.py

CLASSES = ('ahead', 'left', 'right', 'U-turn', 'ahead + right', 'ahead + left', 'left + U-turn',
           'crosswalk', 'bicycle crossing', 'stop line', 'ahead + U-turn')  # = Roadmarks/to_coco.py CLASSES
metainfo = dict(classes=CLASSES)
data_root = WS + '/roadmark_dataset_202/'
IMG_SCALE = (1920, 1536)  # native undistorted size: distant arrows are only a few px tall -> test at full size
CROP = (1280, 1024)       # train on random crops of randomly scaled images

model = dict(
    backbone=dict(norm_cfg=dict(type='BN')),  # single GPU: no SyncBN
    neck=dict(norm_cfg=dict(type='BN')),
    bbox_head=dict(type='RTMDetInsSepBNHeadIgnore', num_classes=len(CLASSES), norm_cfg=dict(type='BN')),
    test_cfg=dict(nms_pre=1000, score_thr=0.05, max_per_img=100))

# NO RandomFlip (left arrow <-> right arrow). No Mosaic / MixUp: MixUp blends two road scenes and Mosaic
# breaks the road geometry; with 56k images and 20 epochs plain scale + crop + colour is enough.
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True, poly2mask=False),
    dict(type='RandomResize', scale=IMG_SCALE, ratio_range=(0.5, 1.5), keep_ratio=True),
    dict(type='RandomCrop', crop_size=CROP, recompute_bbox=True, allow_negative_crop=True),
    dict(type='YOLOXHSVRandomAug'),  # brightness / colour: weather and time of day
    dict(type='Pad', size=CROP, pad_val=dict(img=(114, 114, 114))),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1)),
    dict(type='PackDetInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=IMG_SCALE, keep_ratio=True),
    dict(type='Pad', size=IMG_SCALE, pad_val=dict(img=(114, 114, 114))),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')),
]

train_dataloader = dict(
    batch_size=16, num_workers=12, batch_sampler=None,
    dataset=dict(data_root=data_root, metainfo=metainfo, ann_file='annotations/train.json',
                 data_prefix=dict(img=''), pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=4, num_workers=6,
    dataset=dict(data_root=data_root, metainfo=metainfo, ann_file='annotations/val.json',
                 data_prefix=dict(img=''), pipeline=test_pipeline))
test_dataloader = val_dataloader
val_evaluator = dict(ann_file=data_root + 'annotations/val.json', metric=['bbox', 'segm'], classwise=True)
test_evaluator = val_evaluator

# 56k images / batch 16 = ~3.5k iterations per epoch; 20 epochs: warm-up, flat lr, cosine decay over the second half
max_epochs = 20
base_lr = 0.001  # fine-tuning (COCO recipe: 0.004 for batch 256 from ImageNet)
train_cfg = dict(max_epochs=max_epochs, val_interval=2, dynamic_intervals=[(max_epochs - 4, 1)])
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-3, by_epoch=False, begin=0, end=1000),
    dict(type='CosineAnnealingLR', eta_min=base_lr * 0.05, begin=max_epochs // 2, end=max_epochs,
         T_max=max_epochs // 2, by_epoch=True, convert_to_iter_based=True),
]
optim_wrapper = dict(type='AmpOptimWrapper', optimizer=dict(lr=base_lr))
auto_scale_lr = dict(enable=False)
custom_hooks = [dict(type='EMAHook', ema_type='ExpMomentumEMA', momentum=0.0002, update_buffers=True, priority=49)]

default_hooks = dict(checkpoint=dict(interval=1, max_keep_ckpts=3, save_best='coco/segm_mAP'),
                     logger=dict(interval=100))
load_from = WS + '/mmdetection/checkpoints/rtmdet-ins_s_8xb32-300e_coco_20221121_212604-fdc5d7ec.pth'  # COCO weights
work_dir = WS + '/work_dirs/rtmdet-ins_s_roadmark'
