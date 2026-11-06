#!/usr/bin/env python
"""Run the road-marking model on your own front-camera images. No dataset or HD map needed.

Input: a folder of images (*.jpg / *.png), a glob, or a video file.
  - default: images are already UNDISTORTED front images (fisheye, alpha=0, 1920x1536, like the training data)
  - --raw --prep <dir>: images are RAW fisheye front images; they are undistorted first with the camera calibration
    in <dir> (*__camera_config.json; needs the MotifE2E_Demo repo next to this repo)
Output (--out):
  predictions.json   per image: class, score, bbox [x0, y0, x1, y1], mask polygons (undistorted pixels)
  vis/*.jpg          annotated images        (--vis)
  predictions.mp4    annotated video         (--video, frames in name order, --fps)

usage:
  python Roadmarks/predict.py --ckpt checkpoints/rtmdet-ins_s_roadmark_v1.pth --input /path/to/images --out out/ --vis
"""
import argparse, glob, json, os, sys
from os.path import abspath, basename, dirname, isdir, join, splitext

import cv2
import numpy as np
import torch

HERE = dirname(abspath(__file__))
sys.path.insert(0, HERE)
import rtmdet_ignore  # noqa: E402,F401  registers the custom head
from mmdet.apis import inference_detector, init_detector  # noqa: E402

rng = np.random.default_rng(0)
PALETTE = [tuple(int(c) for c in rng.integers(60, 255, 3)) for _ in range(20)]
IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp')


def frames(src):
    """yields (name, BGR image)"""
    if isdir(src) or any(ch in src for ch in '*?['):
        files = sorted(glob.glob(join(src, '*')) if isdir(src) else glob.glob(src))
        for f in files:
            if f.lower().endswith(IMG_EXT):
                yield basename(f), cv2.imread(f)
    else:
        cap = cv2.VideoCapture(src)
        i = 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            yield f'{splitext(basename(src))[0]}_{i:06d}.jpg', img
            i += 1


def draw(img, preds, names):
    ov = img.copy()
    for p in preds:
        ov[p['mask']] = PALETTE[p['label']]
    out = cv2.addWeighted(img, 0.5, ov, 0.5, 0)
    for p in preds:
        x0, y0, x1, y1 = map(int, p['bbox'])
        col = PALETTE[p['label']]
        cv2.rectangle(out, (x0, y0), (x1, y1), col, 6)
        t = f"{names[p['label']]} {p['score']:.2f}"
        cv2.putText(out, t, (x0, max(30, y0 - 8)), 0, 1.1, (0, 0, 0), 7)
        cv2.putText(out, t, (x0, max(30, y0 - 8)), 0, 1.1, col, 3)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--config', default=join(HERE, 'configs', 'rtmdet-ins_s_roadmark.py'))
    ap.add_argument('--input', required=True, help='image folder, glob ("dir/*.jpg") or video file')
    ap.add_argument('--out', required=True)
    ap.add_argument('--score', type=float, default=0.3, help='keep predictions with score >= this')
    ap.add_argument('--raw', action='store_true', help='input is raw fisheye: undistort first (needs --prep)')
    ap.add_argument('--prep', help='folder with *__camera_config.json (only with --raw)')
    ap.add_argument('--vis', action='store_true', help='write annotated images to <out>/vis/')
    ap.add_argument('--video', action='store_true', help='write <out>/predictions.mp4')
    ap.add_argument('--fps', type=float, default=10)
    ap.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args()

    maps = None
    if a.raw:
        assert a.prep, '--raw needs --prep <folder with *__camera_config.json>'
        from build_roadmark_dataset import front_calibration
        maps = front_calibration(abspath(a.prep))['undist_map']
    os.makedirs(a.out, exist_ok=True)
    if a.vis:
        os.makedirs(join(a.out, 'vis'), exist_ok=True)
    model = init_detector(a.config, a.ckpt, device=a.device)
    names = model.dataset_meta['classes']
    results, vw = [], None
    for i, (name, img) in enumerate(frames(a.input)):
        if img is None:
            continue
        if maps is not None:
            img = cv2.remap(img, maps[0], maps[1], interpolation=cv2.INTER_LINEAR)
        r = inference_detector(model, img).pred_instances
        r = r[r.scores >= a.score]
        preds = [{'label': int(l), 'score': float(s), 'bbox': b, 'mask': m}
                 for l, s, b, m in zip(r.labels.tolist(), r.scores.tolist(), r.bboxes.cpu().numpy().tolist(),
                                       r.masks.cpu().numpy())]
        jp = []
        for p in preds:
            cs, _ = cv2.findContours(p['mask'].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            jp.append({'class': names[p['label']], 'score': round(p['score'], 4),
                       'bbox_xyxy': [round(v, 1) for v in p['bbox']],
                       'mask_polygons': [c.reshape(-1).tolist() for c in cs if len(c) >= 3]})
        results.append({'image': name, 'width': img.shape[1], 'height': img.shape[0], 'predictions': jp})
        if a.vis or a.video:
            vis = draw(img, preds, names)
            if a.vis:
                cv2.imwrite(join(a.out, 'vis', splitext(name)[0] + '.jpg'), vis)
            if a.video:
                if vw is None:
                    h, w = vis.shape[:2]
                    vw = cv2.VideoWriter(join(a.out, 'predictions.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), a.fps, (w, h))
                vw.write(vis)
        if (i + 1) % 100 == 0:
            print(f'{i + 1} images', flush=True)
    if vw is not None:
        vw.release()
    json.dump({'checkpoint': basename(a.ckpt), 'classes': list(names), 'score_thr': a.score,
               'note': 'pixels of the undistorted front image; bbox = [x0, y0, x1, y1]; '
                       'mask_polygons = flat [x, y, x, y, ...] lists',
               'images': results}, open(join(a.out, 'predictions.json'), 'w'))
    print(f'{len(results)} images -> {a.out}')


if __name__ == '__main__':
    main()
