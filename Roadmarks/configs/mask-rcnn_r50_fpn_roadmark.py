# Mask R-CNN R50-FPN, COCO-pretrained, fine-tuned on the HD-map road-marking dataset (roadmark_dataset_202).
# run from mmdetection/:  python tools/train.py ../Roadmarks/configs/mask-rcnn_r50_fpn_roadmark.py
_base_ = '../../mmdetection/configs/mask_rcnn/mask-rcnn_r50_fpn_2x_coco.py'

CLASSES = ('ahead', 'left', 'right', 'U-turn', 'ahead + right', 'ahead + left', 'left + U-turn',
           'crosswalk', 'bicycle crossing', 'stop line', 'ahead + U-turn')  # = Roadmarks/to_coco.py CLASSES (same order)
metainfo = dict(classes=CLASSES)
data_root = '/home/cedric/Downloads/lanes/lane_analysis/data/2/roadmark_dataset_202/'
IMG_SCALE = (1920, 1536)  # native undistorted size: distant arrows are only a few px tall

model = dict(
    roi_head=dict(bbox_head=dict(num_classes=len(CLASSES)), mask_head=dict(num_classes=len(CLASSES))),
    # iscrowd=1 annotations (marks too hidden / far / rare) are ignore regions: anchors and proposals that mostly
    # lie inside them are neither positive nor negative (mmdet's default -1 would treat them as background)
    train_cfg=dict(rpn=dict(assigner=dict(ignore_iof_thr=0.5)),
                   rcnn=dict(assigner=dict(ignore_iof_thr=0.5))))

# NO RandomFlip: flipping turns a left arrow into a right arrow (and moves traffic to the wrong side).
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=IMG_SCALE, keep_ratio=True),
    dict(type='PhotoMetricDistortion'),  # brightness / contrast / colour: weather and time of day
    dict(type='PackDetInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=IMG_SCALE, keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')),
]

train_dataloader = dict(
    batch_size=4, num_workers=8,
    dataset=dict(data_root=data_root, metainfo=metainfo, ann_file='annotations/train.json',
                 data_prefix=dict(img=''), pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=2, num_workers=4,
    dataset=dict(data_root=data_root, metainfo=metainfo, ann_file='annotations/val.json',
                 data_prefix=dict(img=''), pipeline=test_pipeline))
test_dataloader = val_dataloader
val_evaluator = dict(ann_file=data_root + 'annotations/val.json', metric=['bbox', 'segm'], classwise=True)
test_evaluator = val_evaluator

# ~56k train images / batch 4 = ~14k iterations per epoch -> 5 epochs (~7 h), lr x0.1 after epochs 3 and 4
max_epochs = 5
train_cfg = dict(max_epochs=max_epochs, val_interval=1)
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=1000),
    dict(type='MultiStepLR', begin=0, end=max_epochs, by_epoch=True, milestones=[3, 4], gamma=0.1),
]
optim_wrapper = dict(type='AmpOptimWrapper', optimizer=dict(lr=0.005))  # 0.02 is for batch 16 -> 0.005 for batch 4
auto_scale_lr = dict(enable=False)

default_hooks = dict(checkpoint=dict(interval=1, max_keep_ckpts=3, save_best='coco/segm_mAP'),
                     logger=dict(interval=100))
load_from = '/home/cedric/Downloads/lanes/lane_analysis/data/2/mmdetection/checkpoints/' \
            'mask_rcnn_r50_fpn_2x_coco_bbox_mAP-0.392__segm_mAP-0.354_20200505_003907-3e542a40.pth'
work_dir = '/home/cedric/Downloads/lanes/lane_analysis/data/2/work_dirs/mask-rcnn_r50_fpn_roadmark'
