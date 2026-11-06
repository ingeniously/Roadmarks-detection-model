#!/usr/bin/env python3
"""roadmark_dataset_202/labels/*.json -> COCO instance-segmentation json for MMDetection.

  annotations/train.json, annotations/val.json   (file_name relative to the dataset root)

Per mark of a kept frame:
  - usable (same rule as build_roadmark_dataset.py, thresholds read from each label file) and in CLASSES
        -> normal instance; mask = only the visible part (inside the frame, above the bonnet, not behind a vehicle)
  - otherwise but partly visible (too hidden / too far / too small, or a rare class)
        -> iscrowd=1 = ignore region: the model is neither rewarded nor punished there
  - stop lines (open polylines) become a strip ~0.4 m wide on the road (width in px from the flat-ground geometry)

Split: whole days go to val (--val-dates) so near-identical frames never leak between train and val.
--train-every / --val-every subsample in time (consecutive frames at 10 Hz are almost the same).

usage: python Roadmarks/to_coco.py --data roadmark_dataset_202 [--val-dates 20260910] [--train-every 2] [--val-every 5]
"""
import argparse, glob, json, os, sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from os.path import abspath, dirname, join

import cv2
import numpy as np

HERE = dirname(abspath(__file__))
sys.path.insert(0, HERE)
from build_roadmark_dataset import PARTIAL_OK, front_calibration  # noqa: E402
from hdmap_gt import bonnet_y  # noqa: E402

CLASSES = ['ahead', 'left', 'right', 'U-turn', 'ahead + right', 'ahead + left', 'left + U-turn',
           'crosswalk', 'bicycle crossing', 'stop line', 'ahead + U-turn']
RARE = {'ahead + U-turn'}  # frames showing a rare class are never dropped by --train-every / --val-every
MERGE = {'raised crosswalk': 'crosswalk'}
STOP_LINE_WIDTH_M = 0.4
CAM_HEIGHT_M = 1.3195  # CAM_FRONT tz above the ego ground plane (camera_config extrinsic)

_VALID = None


def valid_mask(prep, W, H):
    """undistorted-image pixels that show real image content above the bonnet"""
    global _VALID
    if _VALID is None:
        cal = front_calibration(prep)
        raw_ok = (np.arange(H)[:, None] < bonnet_y(np.arange(W))[None, :]).astype(np.uint8)
        _VALID = cv2.remap(raw_ok, *cal['undist_map'], interpolation=cv2.INTER_NEAREST, borderValue=0)
    return _VALID


def mark_mask(m, K, shape):
    mask = np.zeros(shape, np.uint8)
    p = np.asarray(m['pixels'], np.float64)
    if m['closed']:
        cv2.fillPoly(mask, [np.round(p).astype(np.int32)], 1)
        return mask
    fy, cy = K[1][1], K[1][2]
    for a, b in zip(p[:-1], p[1:]):  # strip around the polyline, half width from ground geometry
        ha, hb = (max(2.0, 0.5 * STOP_LINE_WIDTH_M * max(q[1] - cy, 0) ** 2 / (fy * CAM_HEIGHT_M)) for q in (a, b))
        quad = np.array([[a[0], a[1] - ha], [b[0], b[1] - hb], [b[0], b[1] + hb], [a[0], a[1] + ha]])
        cv2.fillPoly(mask, [np.round(quad).astype(np.int32)], 1)
    return mask


def to_ann(mask):
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segs = [c.reshape(-1).astype(float).tolist() for c in cs if len(c) >= 3 and cv2.contourArea(c) >= 4]
    if not segs:
        return None
    ys, xs = np.nonzero(mask)
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    return {'segmentation': segs, 'bbox': [int(x0), int(y0), int(x1 - x0), int(y1 - y0)], 'area': int(mask.sum())}


def convert_session(task):
    path, root, prep, every = task
    L = json.load(open(path))
    sel = L['selection']
    W, H = L['image_size']
    valid = valid_mask(prep, W, H)
    images, anns, cnt = [], [], Counter()
    for i, f in enumerate(L['frames']):
        rare = any(m['name'] in RARE and m['distance_m'] <= sel['max_dist_m'] and m['visible'] > 0 for m in f['marks'])
        if i % every and not rare:
            continue
        img = {'file_name': f['image'], 'width': W, 'height': H, 'session': L['session'], 'frame': f['frame']}
        fa = []
        seen = valid  # real image area minus the road hidden by vehicles / pedestrians (anno-3DOD)
        if f.get('hidden'):
            hid = np.zeros((H, W), np.uint8)
            cv2.fillPoly(hid, [np.array(p, np.int32) for p in f['hidden']], 1)
            seen = valid & (1 - hid)
        for m in f['marks']:
            name = MERGE.get(m['name'], m['name'])
            usable = m['distance_m'] <= sel['max_dist_m'] and (
                m['visible'] >= sel['min_visible'] or (m['name'] in PARTIAL_OK and (m['visible_px'] or 0) >=
                (sel['or_min_visible_area_px'] if m['closed'] else sel['or_min_visible_line_px'])))
            if not (m['visible_px'] or m['visible'] > 0):
                continue  # completely hidden / outside: nothing to learn or ignore
            a = to_ann(mark_mask(m, L['K_undist'], (H, W)) & seen)
            if a is None:
                continue
            crowd = not (usable and name in CLASSES)
            a.update(category_id=CLASSES.index(name) + 1 if name in CLASSES else 1, iscrowd=int(crowd))
            fa.append(a)
            cnt[('ignore' if crowd else 'inst', name)] += 1
        images.append((img, fa))
    return L['session'], images, cnt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True, help='dataset root (contains labels/ and images/)')
    ap.add_argument('--prep', default=join(HERE, '..', 'prep'), help='folder with the vehicle camera/lidar config')
    ap.add_argument('--val-dates', nargs='+', default=['20260910'])
    ap.add_argument('--force-train', nargs='*', default=['20260910_074942', '20260910_075142'],
                    help='sessions of a val date that go to train anyway (they hold most "ahead + U-turn" frames)')
    ap.add_argument('--train-every', type=int, default=2)
    ap.add_argument('--val-every', type=int, default=5)
    ap.add_argument('--workers', type=int, default=12)
    a = ap.parse_args()

    files = sorted(glob.glob(join(a.data, 'labels', '*.json')))
    def split_of(p):
        s = os.path.basename(p)[:-5]
        return 'val' if s[:8] in a.val_dates and s not in a.force_train else 'train'
    tasks = [(p, a.data, abspath(a.prep), a.val_every if split_of(p) == 'val' else a.train_every) for p in files]
    out = {s: {'images': [], 'annotations': [], 'categories': [{'id': i + 1, 'name': c} for i, c in enumerate(CLASSES)]}
           for s in ('train', 'val')}
    counts, sessions = defaultdict(Counter), defaultdict(int)
    with ProcessPoolExecutor(a.workers) as ex:
        for p, (session, images, cnt) in zip(files, ex.map(convert_session, tasks)):
            s = split_of(p)
            sessions[s] += 1
            counts[s].update(cnt)
            for img, fa in images:
                img['id'] = len(out[s]['images']) + 1
                out[s]['images'].append(img)
                for ann in fa:
                    ann.update(id=len(out[s]['annotations']) + 1, image_id=img['id'])
                    out[s]['annotations'].append(ann)

    os.makedirs(join(a.data, 'annotations'), exist_ok=True)
    for s, d in out.items():
        json.dump(d, open(join(a.data, 'annotations', s + '.json'), 'w'))
        print(f'\n{s}: {sessions[s]} sessions, {len(d["images"])} images, '
              f'{sum(not x["iscrowd"] for x in d["annotations"])} instances, '
              f'{sum(x["iscrowd"] for x in d["annotations"])} ignore regions')
        for c in CLASSES:
            print(f'  {c:18s} {counts[s][("inst", c)]:7d}   (ignore {counts[s][("ignore", c)]})')
        rare = {k[1]: v for k, v in counts[s].items() if k[0] == 'ignore' and k[1] not in CLASSES}
        if rare:
            print('  not in CLASSES -> ignore:', rare)


if __name__ == '__main__':
    main()
