#!/usr/bin/env python
"""Side-by-side check on validation images: left = HD-map projected label, right = model prediction.

usage (env mmdet, from anywhere):
  python Roadmarks/vis_val.py --ckpt work_dirs/mask-rcnn_r50_fpn_roadmark/best_coco_segm_mAP_epoch_1.pth --n 40 --score 0.5
  python Roadmarks/vis_val.py --ckpt work_dirs/rtmdet-ins_s_roadmark/best_coco_segm_mAP_epoch_2.pth --device cpu
output: <ckpt dir>/val_vis/<ckpt name>/NNN_<session>_f<frame>.jpg
"""
import argparse, json, os, sys
from collections import defaultdict
from os.path import abspath, basename, dirname, join, splitext

import cv2
import numpy as np
import torch
from mmdet.apis import inference_detector, init_detector

# torch >= 2.6 loads with weights_only=True, which rejects the mmengine objects in our own training checkpoints
_load = torch.load
torch.load = lambda *a, **k: _load(*a, **{**k, 'weights_only': False})

HERE = dirname(abspath(__file__))
sys.path.insert(0, HERE)
import rtmdet_ignore  # noqa: E402,F401  registers the RTMDet head used in configs/rtmdet-ins_s_roadmark.py


def config_for(ckpt):
    """config = Roadmarks/configs/<work_dir name>.py (work_dirs/<name>/xxx.pth), else the RTMDet config"""
    c = join(HERE, 'configs', basename(dirname(abspath(ckpt))) + '.py')
    return c if os.path.isfile(c) else join(HERE, 'configs', 'rtmdet-ins_s_roadmark.py')
DATA = join(HERE, '..', 'roadmark_dataset_202')
rng = np.random.default_rng(0)
PALETTE = [tuple(int(c) for c in rng.integers(60, 255, 3)) for _ in range(20)]


def put(img, text, xy, col):
    x, y = int(xy[0]), max(25, int(xy[1]) - 6)
    cv2.putText(img, text, (x, y), 0, 0.8, (0, 0, 0), 4)
    cv2.putText(img, text, (x, y), 0, 0.8, col, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--n', type=int, default=40)
    ap.add_argument('--score', type=float, default=0.5)
    ap.add_argument('--device', default='cuda:0', help="'cpu' while a training run fills the GPU")
    ap.add_argument('--config', default=None, help='default: Roadmarks/configs/<work_dir name>.py')
    a = ap.parse_args()
    a.config = a.config or config_for(a.ckpt)

    val = json.load(open(join(DATA, 'annotations', 'val.json')))
    cats = {c['id']: c['name'] for c in val['categories']}
    anns = defaultdict(list)
    for x in val['annotations']:
        anns[x['image_id']].append(x)
    # frames with at least one real instance, spread evenly over the validation set
    ims = [i for i in val['images'] if any(not x['iscrowd'] for x in anns[i['id']])]
    ims = [ims[k] for k in np.linspace(0, len(ims) - 1, a.n).astype(int)]

    model = init_detector(a.config, a.ckpt, device=a.device)
    names = model.dataset_meta['classes']
    out = join(dirname(abspath(a.ckpt)), 'val_vis', splitext(basename(a.ckpt))[0])
    os.makedirs(out, exist_ok=True)

    for k, im in enumerate(ims):
        img = cv2.imread(join(DATA, im['file_name']))
        gt, pr = img.copy(), img.copy()
        ov = gt.copy()
        for x in anns[im['id']]:
            col = (150, 150, 150) if x['iscrowd'] else (0, 220, 0)
            for s in x['segmentation']:
                cv2.fillPoly(ov, [np.array(s).reshape(-1, 2).astype(np.int32)], col)
        gt = cv2.addWeighted(gt, 0.5, ov, 0.5, 0)
        for x in anns[im['id']]:
            if not x['iscrowd']:
                put(gt, cats[x['category_id']], x['bbox'][:2], (255, 255, 255))
        put(gt, f'HD-map label  {im["session"]} f{im["frame"]}', (20, 50), (0, 255, 255))

        r = inference_detector(model, img).pred_instances
        keep = r.scores > a.score
        ov = pr.copy()
        for m, l in zip(r.masks[keep].cpu().numpy(), r.labels[keep].tolist()):
            ov[m] = PALETTE[l]
        pr = cv2.addWeighted(pr, 0.5, ov, 0.5, 0)
        for b, l, s in zip(r.bboxes[keep].cpu().numpy(), r.labels[keep].tolist(), r.scores[keep].tolist()):
            cv2.rectangle(pr, tuple(b[:2].astype(int)), tuple(b[2:].astype(int)), PALETTE[l], 2)
            put(pr, f'{names[l]} {s:.2f}', b[:2], PALETTE[l])
        put(pr, f'prediction (score > {a.score})', (20, 50), (0, 255, 255))

        both = np.hstack([gt, pr])[:1180]  # drop the bonnet rows
        cv2.imwrite(join(out, f'{k:03d}_{im["session"]}_f{im["frame"]}.jpg'),
                    cv2.resize(both, (both.shape[1] // 2, both.shape[0] // 2)))
    print('wrote', len(ims), 'images to', out)


if __name__ == '__main__':
    main()
