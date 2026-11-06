#!/usr/bin/env python
"""Mark-level ("found" / "false alarm") accuracy from the outputs of infer_sessions.py.

Found:       a real marking (one HD-map id, usable in >= 1 frame) is found when the model predicts the right class on it
             (mask IoU >= --iou, both masks dilated by --tol px) in at least --min-hits frames.
Wrong type:  the marking was only hit with another class of the same group (e.g. 'left' predicted on 'ahead + left').
False alarm: a prediction (score >= --score) that overlaps no HD-map marking at all (any class, incl. ignore regions,
             IoU < 0.05); consecutive-frame predictions of the same group at the same place form one alarm (event),
             and an event must last >= --min-frames frames.
Groups: road marks (all arrows), crosswalk, bicycle crossing, stop line.

usage (env mmdet): python Roadmarks/event_eval.py --dir work_dirs/infer --sessions-root <server>/sessions
"""
import argparse, glob, json, os, sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from os.path import abspath, dirname, join

import cv2
import numpy as np
from pycocotools import mask as mU

HERE = dirname(abspath(__file__))
sys.path.insert(0, HERE)
from infer_sessions import ROOT, TRAIN_SESSIONS, WORK, session_gt  # noqa: E402
from to_coco import CLASSES  # noqa: E402

GROUP = {c: 'road marks (arrows)' for c in CLASSES[:7] + ['ahead + U-turn']}
GROUP.update({'crosswalk': 'crosswalk', 'bicycle crossing': 'bicycle crossing', 'stop line': 'stop line'})
GROUPS = ['road marks (arrows)', 'crosswalk', 'stop line', 'bicycle crossing']


def box_iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0


def eval_session(args):
    session, d, a = args
    P = json.load(open(join(d, session, session + '_predictions.json')))
    W, H = P['image_size']
    pred_by_frame = {f['frame']: f['predictions'] for f in P['frames']}
    hits = defaultdict(lambda: defaultdict(int))  # mark id -> class -> frames hit
    marks = {}  # id -> class name of usable marks
    alarms = defaultdict(list)  # group -> open/closed events [first, last, box, n]
    for f, anns in session_gt(session, a.sessions_root, abspath(a.prep)):
        for g in anns:
            if not g['crowd']:
                marks[g['id']] = CLASSES[g['cat']]
        preds = [p for p in pred_by_frame.get(f['frame'], []) if p['score'] >= a.score]
        if not preds:
            continue
        # compare at 1/4 resolution with both masks dilated by --tol px: the HD-map projection can be a few px off,
        # which otherwise gives a thin stop line IoU 0 although the prediction lies on the real line
        k = np.ones((2 * (a.tol // 4) + 1,) * 2, np.uint8)
        small = lambda m: mU.encode(np.asfortranarray(cv2.dilate(cv2.resize(  # noqa: E731
            m.astype(np.uint8), (W // 4, H // 4), interpolation=cv2.INTER_NEAREST), k)))
        rles = []
        for p in preds:
            m = np.zeros((H, W), np.uint8)
            cv2.fillPoly(m, [np.array(q, np.int32).reshape(-1, 2) for q in p['mask_polygons']], 1)
            rles.append(small(m))
        g_rles = [small(g['mask']) for g in anns]
        iou = mU.iou(rles, g_rles, [0] * len(anns)) if anns else np.zeros((len(preds), 0))
        for i, p in enumerate(preds):
            for j, g in enumerate(anns):
                if iou[i, j] >= a.iou and not g['crowd']:
                    hits[g['id']][p['class']] += 1
            if not anns or iou[i].max() < 0.05:  # on no marking at all -> false alarm candidate
                grp, b = GROUP[p['class']], p['bbox_xyxy']
                ev = next((e for e in alarms[grp] if f['frame'] - e[1] <= 2 and box_iou(e[2], b) > 0.1), None)
                if ev:
                    ev[1], ev[2], ev[3] = f['frame'], b, ev[3] + (ev[1] != f['frame'])
                else:
                    alarms[grp].append([f['frame'], f['frame'], b, 1])
    out = {g: {'marks': 0, 'found': 0, 'wrong_type': 0, 'false_alarms': 0} for g in GROUPS}
    for mid, c in marks.items():
        o = out[GROUP[c]]
        o['marks'] += 1
        if hits[mid].get(c, 0) >= a.min_hits:
            o['found'] += 1
        elif any(n >= a.min_hits and GROUP[k] == GROUP[c] for k, n in hits[mid].items()):
            o['wrong_type'] += 1
    for g in GROUPS:
        ev = [e for e in alarms[g] if e[3] >= a.min_frames]
        out[g]['false_alarms'] = len(ev)
        out[g]['false_alarm_frames'] = [[e[0], e[1]] for e in ev]  # first / last frame, to look them up in the video
    return session, out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dir', default=join(WORK, 'infer'), help='--out of infer_sessions.py')
    ap.add_argument('--sessions-root', default=os.environ.get('SESSIONS_ROOT'), required='SESSIONS_ROOT' not in os.environ,
                    help='mounted server sessions folder (or env SESSIONS_ROOT)')
    ap.add_argument('--prep', default=join(ROOT, 'prep'))
    ap.add_argument('--score', type=float, default=0.5)
    ap.add_argument('--iou', type=float, default=0.3, help='mask IoU for a hit (HD-map labels can be shifted)')
    ap.add_argument('--tol', type=int, default=16, help='px tolerance for HD-map projection offsets')
    ap.add_argument('--min-hits', type=int, default=1)
    ap.add_argument('--min-frames', type=int, default=3, help='a false alarm must persist this many frames')
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    sessions = sorted(os.path.basename(p)[:-len('_predictions.json')]
                      for p in glob.glob(join(a.dir, '*', '*_predictions.json')))
    res = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for s, o in ex.map(eval_session, [(s, a.dir, a) for s in sessions]):
            res[s] = o
            print(s, {g: f"{v['found']}/{v['marks']} fa={v['false_alarms']}" for g, v in o.items()}, flush=True)
    tot = {}
    for split in ('val', 'all'):
        t = {g: defaultdict(int) for g in GROUPS}
        for s, o in res.items():
            if split == 'val' and s in TRAIN_SESSIONS:
                continue
            for g in GROUPS:
                for k, v in o[g].items():
                    if k != 'false_alarm_frames':
                        t[g][k] += v
        tot[split] = {g: dict(v, found_pct=round(100 * v['found'] / max(v['marks'], 1), 1)) for g, v in t.items()}
    json.dump({'settings': vars(a), 'total': tot, 'sessions': res}, open(join(a.dir, 'event_summary.json'), 'w'),
              indent=1)
    n_val = sum(s not in TRAIN_SESSIONS for s in res)
    print(f'\n{n_val} val sessions (training sessions excluded):')
    for g, v in tot['val'].items():
        print(f"  {g:20s} found {v['found_pct']:5.1f}%  ({v['found']}/{v['marks']}, wrong type {v['wrong_type']}),"
              f"  {v['false_alarms']} false alarms")


if __name__ == '__main__':
    main()
