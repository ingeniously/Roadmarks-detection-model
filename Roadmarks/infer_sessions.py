#!/usr/bin/env python
"""Run the trained road-marking model on EVERY front frame of whole sessions and, per session, write
  <out>/<session>/<session>.mp4               side by side: left = GT (HD-map projection), right = prediction
  <out>/<session>/<session>_predictions.json  per-frame predictions (class, score, box, mask polygons)
  <out>/<session>/<session>_metrics.json      accuracy against the HD-map GT of all frames
and <out>/summary.csv / summary.json with one row per session.

The GT of every frame is made exactly like the training labels (build_roadmark_dataset.py + to_coco.py):
visible part only, marks too hidden / too far / rare -> ignore region (grey), not counted either way.

usage (env mmdet):
  python Roadmarks/infer_sessions.py --ckpt <checkpoint.pth> --sessions '20260910_*' --sessions-root <server>/sessions
"""
import argparse, contextlib, csv, fnmatch, glob, io, json, math, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from os.path import abspath, dirname, join

import cv2
import numpy as np
import torch
from pycocotools import mask as mU
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

_load = torch.load  # torch >= 2.6 weights_only=True rejects the mmengine objects in our checkpoints
torch.load = lambda *a, **k: _load(*a, **{**k, 'weights_only': False})
from mmdet.apis import inference_detector, init_detector  # noqa: E402

HERE = dirname(abspath(__file__))
sys.path.insert(0, HERE)
from build_roadmark_dataset import PARTIAL_OK, S, Occlusion, front_calibration, is_target  # noqa: E402
from to_coco import CLASSES, MERGE, mark_mask, valid_mask  # noqa: E402
import rtmdet_ignore  # noqa: E402,F401  registers the custom RTMDet head

ROOT = abspath(join(HERE, '..'))  # workspace
WORK = join(ROOT, 'work_dirs')
CFG = join(HERE, 'configs', 'rtmdet-ins_s_roadmark.py')
DATA = join(ROOT, 'roadmark_dataset_202')  # needs gt/<session>.json (build_roadmark_dataset.py --only gt)
SEL = dict(min_visible=0.5, max_dist_m=30, min_area=4000, min_len=200)  # = dataset selection
TRAIN_SESSIONS = {'20260910_074942', '20260910_075142'}  # moved to train (ahead + U-turn): metrics are optimistic
rng = np.random.default_rng(0)
PALETTE = [tuple(int(c) for c in rng.integers(60, 255, 3)) for _ in range(20)]
CROP_Y = 1180  # drop the bonnet rows in the video


def put(img, text, xy, col, scale=1.0, th=2):
    x, y = int(xy[0]), max(int(32 * scale), int(xy[1]) - 8)
    cv2.putText(img, text, (x, y), 0, scale, (0, 0, 0), th + 4)
    cv2.putText(img, text, (x, y), 0, scale, col, th)


def corner_tag(img, text, col):
    cv2.rectangle(img, (0, 0), (60 + 34 * len(text), 90), (0, 0, 0), -1)
    cv2.putText(img, text, (25, 68), 0, 2.2, col, 5)


# ------------------------------------------------------------------ ground truth for every frame
def session_gt(session, sessions_root, prep):
    """yields (frame record, [ann]) for every frame with an image; ann = dict(cat, crowd, rle, bbox, poly)"""
    cal = front_calibration(prep)
    K, D, new_K = cal['intrinsic_raw'], cal['dist'], cal['intrinsic']
    map1, map2 = cal['undist_map']
    gt = json.load(open(join(DATA, 'gt', session + '.json')))
    W, H = gt['image_size']
    valid = valid_mask(prep, W, H)
    od_files = sorted(glob.glob(join(sessions_root, session, 'annotation', 'anno-3DOD', '*.json')),
                      key=lambda p: os.path.basename(p) != 'anno-3DOD.json')
    od = json.load(open(od_files[0])) if od_files else None
    occ, T_e_lid = None, None
    if od:
        occ = Occlusion(cal, np.array(od['calibration']['CAM_FRONT']['T_parent_child']), map1, map2, W, H)
        T_e_lid = np.array(od['calibration']['LIDAR_TOP']['T_parent_child'])
    Kl = np.round(new_K, 4).tolist()
    for f in gt['frames']:
        if not f['image']:
            continue
        marks = [m for m in f['marks'] if is_target(m) and m['visible'] > 0]
        anns = []
        if marks:
            hidden = None
            if occ:
                objs = (od['frames'].get(str(f['frame'])) or {}).get('objects', [])
                hidden = occ.frame_mask(objs, T_e_lid)
            seen = valid
            if hidden is not None and hidden.any():
                hid = cv2.resize(hidden.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
                seen = valid & (1 - hid)
            for m in marks:
                uv = cv2.fisheye.undistortPoints(np.array(m['pixels_raw'], np.float64).reshape(-1, 1, 2),
                                                 K, D, P=new_K).reshape(-1, 2)
                closed = m['layer'] == 'B3'
                frac, size = occ.visible(uv, closed, hidden) if occ else (m['visible'], None)
                if not (size or frac > 0):
                    continue
                name = MERGE.get(m['name'], m['name'])
                usable = m['distance_m'] <= SEL['max_dist_m'] and (
                    frac >= SEL['min_visible'] or (m['name'] in PARTIAL_OK and (size or 0) >=
                                                   (SEL['min_area'] if closed else SEL['min_len'])))
                mk = mark_mask({'pixels': uv, 'closed': closed}, Kl, (H, W)) & seen
                if mk.sum() < 4:
                    continue
                rle = mU.encode(np.asfortranarray(mk))
                anns.append(dict(id=m['id'], cat=CLASSES.index(name) if name in CLASSES else 0,
                                 crowd=not (usable and name in CLASSES), rle=rle,
                                 bbox=mU.toBbox(rle).tolist(), mask=mk))
        yield f, anns


# ------------------------------------------------------------------ drawing
def draw_gt(img, anns):
    ov = img.copy()
    for a in anns:
        ov[a['mask'].astype(bool)] = (150, 150, 150) if a['crowd'] else (0, 220, 0)
    out = cv2.addWeighted(img, 0.5, ov, 0.5, 0)
    for a in anns:
        if not a['crowd']:
            x, y, w, h = map(int, a['bbox'])
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 3)
            put(out, CLASSES[a['cat']], (x, y), (255, 255, 255), 1.1, 3)
    corner_tag(out, 'GT', (0, 255, 0))
    return out


def draw_pred(img, preds, thr):
    ov = img.copy()
    keep = [p for p in preds if p['score'] >= thr]
    for p in keep:
        ov[p['mask']] = PALETTE[p['cat']]
    out = cv2.addWeighted(img, 0.5, ov, 0.5, 0)
    for p in keep:
        x0, y0, x1, y1 = map(int, p['bbox_xyxy'])
        cv2.rectangle(out, (x0, y0), (x1, y1), PALETTE[p['cat']], 6)  # thick prediction boxes
        put(out, f"{CLASSES[p['cat']]} {p['score']:.2f}", (x0, y0), PALETTE[p['cat']], 1.1, 3)
    corner_tag(out, 'PREDICTION', (0, 255, 255))
    return out


# ------------------------------------------------------------------ metrics
def simple_pr(frames_gt, frames_pred, thr, iou_thr=0.5):
    """precision / recall per class at score >= thr, mask IoU >= iou_thr; predictions mostly (iof >= 0.5)
    inside an ignore region are not counted"""
    tp, fp, fn = np.zeros(len(CLASSES), int), np.zeros(len(CLASSES), int), np.zeros(len(CLASSES), int)
    for anns, preds in zip(frames_gt, frames_pred):
        crowd = [a['rle'] for a in anns if a['crowd']]
        for c in range(len(CLASSES)):
            g = [a['rle'] for a in anns if not a['crowd'] and a['cat'] == c]
            p = sorted([q for q in preds if q['cat'] == c and q['score'] >= thr], key=lambda q: -q['score'])
            used = set()
            if g and p:
                iou = mU.iou([q['rle'] for q in p], g, [0] * len(g))
            for i, q in enumerate(p):
                j = -1
                if g:
                    cand = [(iou[i, k], k) for k in range(len(g)) if k not in used and iou[i, k] >= iou_thr]
                    j = max(cand)[1] if cand else -1
                if j >= 0:
                    used.add(j); tp[c] += 1
                elif crowd and mU.iou([q['rle']], crowd, [1] * len(crowd)).max() >= 0.5:
                    pass
                else:
                    fp[c] += 1
            fn[c] += len(g) - len(used)
    return tp, fp, fn


def coco_eval(frames_gt, frames_pred, W, H):
    images, gts, dts = [], [], []
    for i, (anns, preds) in enumerate(zip(frames_gt, frames_pred), 1):
        images.append({'id': i, 'width': W, 'height': H})
        for a in anns:
            r = dict(a['rle']); r['counts'] = r['counts'].decode()
            gts.append({'id': len(gts) + 1, 'image_id': i, 'category_id': a['cat'] + 1, 'iscrowd': int(a['crowd']),
                        'segmentation': r, 'bbox': a['bbox'], 'area': float(mU.area(a['rle']))})
        for p in preds:
            r = dict(p['rle']); r['counts'] = r['counts'].decode()
            x0, y0, x1, y1 = p['bbox_xyxy']
            dts.append({'image_id': i, 'category_id': p['cat'] + 1, 'score': p['score'], 'segmentation': r,
                        'bbox': [x0, y0, x1 - x0, y1 - y0]})
    res = {}
    if not any(not g['iscrowd'] for g in gts):
        return {'note': 'no usable GT marking in this session'}
    with contextlib.redirect_stdout(io.StringIO()):
        cg = COCO(); cg.dataset = {'images': images, 'annotations': gts,
                                   'categories': [{'id': i + 1, 'name': c} for i, c in enumerate(CLASSES)]}
        cg.createIndex()
        cd = cg.loadRes(dts) if dts else None
        for t in ('bbox', 'segm'):
            if cd is None:
                res[t] = {'mAP': 0.0, 'mAP_50': 0.0}; continue
            E = COCOeval(cg, cd, t); E.evaluate(); E.accumulate(); E.summarize()
            res[t] = {'mAP': round(float(E.stats[0]), 3), 'mAP_50': round(float(E.stats[1]), 3)}
            if t == 'segm':  # per class AP50 (only classes with GT in this session)
                pc = {}
                for k, c in enumerate(CLASSES):
                    pr = E.eval['precision'][0, :, k, 0, -1]
                    if (pr > -1).any():
                        pc[c] = round(float(pr[pr > -1].mean()), 3)
                res['segm_AP50_per_class'] = pc
    return res


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sessions', nargs='+', default=['20260910_*'], help='session names or patterns')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--config', default=CFG)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--sessions-root', default=os.environ.get('SESSIONS_ROOT'), required='SESSIONS_ROOT' not in os.environ,
                    help='mounted server sessions folder (or env SESSIONS_ROOT)')
    ap.add_argument('--prep', default=join(ROOT, 'prep'), help='folder with *__camera_config.json')
    ap.add_argument('--out', default=join(WORK, 'infer'))
    ap.add_argument('--score', type=float, default=0.5, help='threshold for the video and precision/recall')
    ap.add_argument('--json-score', type=float, default=0.3, help='lowest score written to the predictions json')
    ap.add_argument('--fps', type=float, default=10)
    ap.add_argument('--limit', type=int, default=None, help='only the first N frames per session (test)')
    a = ap.parse_args()

    all_s = sorted(os.path.basename(p)[:-5] for p in glob.glob(join(DATA, 'gt', '*.json')))
    sessions = [s for s in all_s if any(fnmatch.fnmatch(s, p) for p in a.sessions)]
    print(f'{len(sessions)} sessions -> {a.out}', flush=True)
    model = init_detector(a.config, a.ckpt, device=a.device)
    cal = front_calibration(abspath(a.prep))
    map1, map2 = cal['undist_map']
    rows = []
    pool = ThreadPoolExecutor(8)

    def load(f, session):
        img = cv2.imread(join(a.sessions_root, session, 'perception_sensor_data', 'cam-rgb-front', f['image']))
        return None if img is None else cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)

    for si, session in enumerate(sessions, 1):
        t0 = time.time()
        od = join(a.out, session)
        os.makedirs(od, exist_ok=True)
        if os.path.isfile(join(od, session + '_metrics.json')):
            rows.append(json.load(open(join(od, session + '_metrics.json'))))
            continue
        g = json.load(open(join(DATA, 'gt', session + '.json')))
        W, H = g['image_size']
        todo = [f for f in g['frames'] if f['image']][:a.limit]  # same frames, same order as session_gt
        futs = [pool.submit(load, f, session) for f in todo]  # images load ahead in 8 threads
        tmp_mp4 = join(od, session + '.tmp.mp4')
        vw = cv2.VideoWriter(tmp_mp4, cv2.VideoWriter_fourcc(*'mp4v'), a.fps, (W, CROP_Y // 2))
        frames_gt, frames_pred, jframes = [], [], []
        for (f, anns), fut in zip(session_gt(session, a.sessions_root, abspath(a.prep)), futs):
            img = fut.result()
            if img is None:
                continue
            r = inference_detector(model, img).pred_instances
            masks = r.masks.cpu().numpy()
            preds = []
            for m, l, s, b in zip(masks, r.labels.tolist(), r.scores.tolist(), r.bboxes.cpu().numpy().tolist()):
                preds.append({'cat': l, 'score': s, 'bbox_xyxy': b, 'mask': m,
                              'rle': mU.encode(np.asfortranarray(m.astype(np.uint8)))})
            vis = np.hstack([draw_gt(img, anns), draw_pred(img, preds, a.score)])[:CROP_Y]
            put(vis, f'{session}  frame {f["frame"]}', (W - 330, CROP_Y - 15), (255, 255, 255), 0.9, 2)
            vw.write(cv2.resize(vis, (W, CROP_Y // 2), interpolation=cv2.INTER_AREA))
            jp = []
            for p in preds:
                if p['score'] < a.json_score:
                    continue
                cs, _ = cv2.findContours(p['mask'].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                jp.append({'class': CLASSES[p['cat']], 'score': round(p['score'], 4),
                           'bbox_xyxy': [round(v, 1) for v in p['bbox_xyxy']],
                           'mask_polygons': [c.reshape(-1).tolist() for c in cs if len(c) >= 3]})
            jframes.append({'frame': f['frame'], 'timestamp': f['timestamp'], 'image': f['image'], 'predictions': jp})
            for p in preds:
                del p['mask']
            for g in anns:
                del g['mask']
            frames_gt.append(anns); frames_pred.append(preds)
        vw.release()
        mp4 = join(od, session + '.mp4')  # re-encode to H.264 so it plays in browsers / VS Code
        if os.system(f'ffmpeg -loglevel error -y -i "{tmp_mp4}" -c:v libx264 -crf 26 -pix_fmt yuv420p "{mp4}"') == 0:
            os.remove(tmp_mp4)
        else:
            os.replace(tmp_mp4, mp4)

        json.dump({'session': session, 'checkpoint': os.path.basename(a.ckpt), 'classes': CLASSES,
                   'image_size': [W, H], 'K_undist': np.round(cal['intrinsic'], 4).tolist(),
                   'note': 'coordinates in the undistorted front image (fisheye, alpha=0); '
                           f'predictions with score >= {a.json_score}; bbox = [x0, y0, x1, y1]; '
                           'mask_polygons = flat [x, y, x, y, ...] lists',
                   'frames': jframes}, open(join(od, session + '_predictions.json'), 'w'))

        tp, fp, fn = simple_pr(frames_gt, frames_pred, a.score)
        coco = coco_eval(frames_gt, frames_pred, W, H)
        T, P, N = tp.sum(), fp.sum(), fn.sum()
        row = {'session': session, 'split': 'train' if session in TRAIN_SESSIONS else 'val',
               'frames': len(frames_gt), 'frames_with_gt': sum(any(not g['crowd'] for g in x) for x in frames_gt),
               'gt_instances': int(T + N), 'pred_instances': int(T + P),
               f'precision@{a.score}': round(T / max(T + P, 1), 3), f'recall@{a.score}': round(T / max(T + N, 1), 3),
               'segm_mAP': coco.get('segm', {}).get('mAP'), 'segm_mAP_50': coco.get('segm', {}).get('mAP_50'),
               'bbox_mAP': coco.get('bbox', {}).get('mAP'), 'bbox_mAP_50': coco.get('bbox', {}).get('mAP_50'),
               'per_class': {c: {'tp': int(tp[k]), 'fp': int(fp[k]), 'fn': int(fn[k]),
                                 'segm_AP50': coco.get('segm_AP50_per_class', {}).get(c)}
                             for k, c in enumerate(CLASSES) if tp[k] + fp[k] + fn[k]},
               'note': coco.get('note', '')}
        json.dump(row, open(join(od, session + '_metrics.json'), 'w'), indent=1)
        rows.append(row)
        print(f'[{si}/{len(sessions)}] {session} {row["frames"]} frames  P={row[f"precision@{a.score}"]} '
              f'R={row[f"recall@{a.score}"]} segm AP50={row["segm_mAP_50"]}  ({time.time() - t0:.0f}s)', flush=True)

    keys = [k for k in rows[0] if k not in ('per_class', 'note')]
    with open(join(a.out, 'summary.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, keys, extrasaction='ignore'); w.writeheader(); w.writerows(rows)
    json.dump(rows, open(join(a.out, 'summary.json'), 'w'), indent=1)
    print('done ->', join(a.out, 'summary.csv'))


if __name__ == '__main__':
    main()
