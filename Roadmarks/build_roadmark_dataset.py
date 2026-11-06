#!/usr/bin/env python3
"""Build a front-camera road-marking dataset (arrows, crosswalks, stop lines) from the sessions of an
e2e_prepare export, keeping only the frames that actually show such a marking.

Per session:
  1. gt     : project the HD map onto every front frame with hdmap_gt.py  -> <out>/gt/<session>.json
              (raw fisheye pixels; same output as hdmap_gt_front.json, written locally, never on the server)
  2. build  : keep frames with >= 1 target mark (visible >= --min-visible, distance <= --max-dist; visible
              accounts for vehicles/pedestrians from anno-3DOD that hide the mark),
              undistort those front images exactly like s3_undistort (fisheye, alpha=0, JPEG q95)
              -> <out>/images/<session>/<image>.jpg
              and convert the mark polygons to undistorted pixels
              -> <out>/labels/<session>.json

Every target mark seen in a kept frame is stored (also faint/far ones, with their `visible` and `distance_m`)
so the training converter can decide what to use and what to ignore. Re-running skips finished work.

usage:
  build_roadmark_dataset.py --prepare /home/cedric/server-1.201.66.58/20261029_185000_e2e_prepare_202sessions \
                            --sessions-root /home/cedric/server-1.201.66.58/sessions \
                            --prep /home/cedric/server-1.201.66.58/prep --out ../roadmark_dataset_202
  options: --only gt|build   --limit N (first N sessions, for testing)   --every N (keep every Nth kept frame)
"""
import argparse, glob, json, math, os, subprocess, sys, time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from os.path import abspath, dirname, isfile, join

import cv2
import numpy as np

HERE = dirname(abspath(__file__))
REPO = abspath(join(HERE, '..', 'MotifE2E_Demo'))

# classes to detect (names as written by hdmap_gt.py)
TARGET_B3 = {'ahead', 'ahead + right', 'ahead + left', 'right', 'left', 'U-turn', 'left + U-turn',
             'ahead + U-turn', 'left + right', 'crosswalk', 'raised crosswalk', 'bicycle crossing'}
TARGET_B2 = {'stop line'}
PARTIAL_OK = {'crosswalk', 'raised crosswalk', 'bicycle crossing', 'stop line'}  # see run_build
JPEG_Q = 95  # = UNDISTORT_JPEG_QUALITY in MotifE2E_Demo/tools/data_pipeline/common/layout.py

_CAL = None


def front_calibration(prep):
    """K, D (fisheye), new_K and remap LUT of CAM_FRONT, computed by the same code as s3_undistort."""
    global _CAL
    if _CAL is None:
        sys.path.insert(0, REPO)
        import tools.data_pipeline.common.layout as L
        from tools.data_pipeline.common.calib import load_calibration
        L.config_search_roots = lambda d: [prep]
        c = load_calibration(prep, 'session', alpha=L.DEFAULT_UNDISTORT_ALPHA)['cameras']['CAM_FRONT']
        _CAL = c
    return _CAL


def is_target(m):
    return m['name'] in TARGET_B3 if m['layer'] == 'B3' else m['name'] in TARGET_B2


# ---------------------------------------------------------------- occlusion (undistorted image, 1/S resolution)
S = 4  # grid step [px]
BOX_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


class Occlusion:
    """Which pixels of the undistorted front image show the road surface, and the 3D boxes that hide it.

    range_map : horizontal distance [m] from the ego origin to where the pixel ray hits the ground plane (inf = sky)
    valid     : pixel shows real image content above the bonnet
    A mark pixel is hidden when its camera ray enters a 3D box (anno-3DOD, all classes) before it reaches
    the road (flat-road assumption: ground = ego z=0 plane).
    """

    def __init__(self, cal, cam2ego, map1, map2, W, H):
        self.K, self.W, self.H = cal['intrinsic'], W, H
        self.R, self.t = cam2ego[:3, :3], cam2ego[:3, 3]
        gw, gh = W // S, H // S
        u, v = np.meshgrid((np.arange(gw) + 0.5) * S, (np.arange(gh) + 0.5) * S)
        rays = np.stack([u, v, np.ones_like(u)], -1) @ np.linalg.inv(self.K).T @ self.R.T  # ego-frame directions
        with np.errstate(divide='ignore', invalid='ignore'):
            s = np.where(rays[..., 2] < -1e-6, -self.t[2] / rays[..., 2], np.inf)
            self.range_map = np.hypot(self.t[0] + s * rays[..., 0], self.t[1] + s * rays[..., 1])
        self.rays, self.s_ground = rays, s
        self.margin = 0.3 / np.linalg.norm(rays, axis=-1)  # 0.3 m along the ray, in ray-parameter units
        sys.path.insert(0, HERE)
        from hdmap_gt import bonnet_y
        raw_ok = (np.arange(H)[:, None] < bonnet_y(np.arange(W))[None, :]).astype(np.uint8) * 255
        und_ok = cv2.remap(raw_ok, map1, map2, interpolation=cv2.INTER_NEAREST, borderValue=0)
        self.valid = cv2.resize(und_ok, (gw, gh), interpolation=cv2.INTER_AREA) > 127

    def frame_mask(self, objects, ego2lidar_T):
        """bool grid: True = road pixel hidden by an object"""
        hidden = np.zeros(self.range_map.shape, bool)
        R_el, t_el = ego2lidar_T[:3, :3], ego2lidar_T[:3, 3]
        for o in objects:
            b = o['box']
            c, s_ = math.cos(b['yaw']), math.sin(b['yaw'])
            half = np.array([b['length'], b['width'], b['height']]) / 2
            R_b = R_el @ np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])  # box -> ego
            c_b = R_el @ [b['x'], b['y'], b['z']] + t_el
            if np.hypot(*c_b[:2]) > 80:
                continue
            loc = np.array([[sx, sy, sz] for sz in (-1, 1) for sx, sy in ((1, 1), (1, -1), (-1, -1), (-1, 1))]) * half
            ego = loc @ R_b.T + c_b
            # region of interest = silhouette of the box part in front of the camera (edges sampled, slightly dilated)
            edge = np.concatenate([ego[a] + (ego[b_] - ego[a]) * np.linspace(0, 1, 12)[:, None] for a, b_ in BOX_EDGES])
            cam = (edge - self.t) @ self.R
            cam = cam[cam[:, 2] > 0.3]
            if len(cam) < 3:
                continue
            uv = cam[:, :2] / cam[:, 2:] * [self.K[0, 0], self.K[1, 1]] + [self.K[0, 2], self.K[1, 2]]
            hull = cv2.convexHull(np.clip(uv / S, -1e4, 1e4).astype(np.float32)).round().astype(np.int32)
            roi = np.zeros(hidden.shape, np.uint8)
            cv2.fillPoly(roi, [hull], 1)
            roi = cv2.dilate(roi, np.ones((5, 5), np.uint8)).astype(bool)
            if not roi.any():
                continue
            # exact test: does the pixel ray enter the box before it hits the road? (slab method, box frame)
            d = self.rays[roi] @ R_b
            d = np.where(np.abs(d) < 1e-9, 1e-9, d)
            o_ = R_b.T @ (self.t - c_b)
            t1, t2 = (-half - o_) / d, (half - o_) / d
            tmin, tmax = np.minimum(t1, t2).max(1), np.maximum(t1, t2).min(1)
            hit = (tmax >= np.maximum(tmin, 0)) & (tmin < self.s_ground[roi] - self.margin[roi])
            hidden[roi] |= hit
        return hidden

    def visible(self, pixels, closed, hidden):
        """(fraction, size) of the mark that is really visible: area [px^2] for polygons, length [px] for lines"""
        p = np.asarray(pixels, float)
        if closed:
            total = cv2.contourArea(p.astype(np.float32)) / S ** 2
            if total < 1:
                return 0.0, 0
            m = np.zeros(hidden.shape, np.uint8)
            cv2.fillPoly(m, [np.round(p / S).astype(np.int32)], 1)
            seen = int((m.astype(bool) & self.valid & ~hidden).sum())
            return min(1.0, seen / total), seen * S * S
        seg = [a + (b - a) * k for a, b in zip(p[:-1], p[1:]) for k in np.linspace(0, 1, max(2, int(np.linalg.norm(b - a) / 2)))]
        q = (np.array(seg) / S).astype(int)
        gh, gw = hidden.shape
        ok = (q[:, 0] >= 0) & (q[:, 0] < gw) & (q[:, 1] >= 0) & (q[:, 1] < gh)
        qi = q[ok]
        frac = float((self.valid[qi[:, 1], qi[:, 0]] & ~hidden[qi[:, 1], qi[:, 0]]).sum() / len(q))
        return frac, int(frac * np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def run_gt(session, sessions_root, cam_cfg, out_dir):
    dst = join(out_dir, 'gt', session + '.json')
    if isfile(dst):
        return session, 'skip'
    tmp = dst + '.tmp'
    r = subprocess.run([sys.executable, join(HERE, 'hdmap_gt.py'), join(sessions_root, session),
                        '--cam-config', cam_cfg, '--out', tmp], capture_output=True, text=True)
    if r.returncode != 0:
        return session, 'FAIL: ' + (r.stderr.strip().splitlines() or ['?'])[-1]
    os.replace(tmp, dst)
    return session, 'ok'


def run_build(session, sessions_root, prep, out_dir, min_vis, max_dist, every, min_area, min_len):
    dst_lbl = join(out_dir, 'labels', session + '.json')
    if isfile(dst_lbl):
        return session, json.load(open(dst_lbl))['stats']
    cal = front_calibration(prep)
    K, D, new_K = cal['intrinsic_raw'], cal['dist'], cal['intrinsic']
    map1, map2 = cal['undist_map']
    gt = json.load(open(join(out_dir, 'gt', session + '.json')))
    src_dir = join(sessions_root, session, 'perception_sensor_data', 'cam-rgb-front')
    img_dir = join(out_dir, 'images', session)
    os.makedirs(img_dir, exist_ok=True)

    # anno-3DOD.json, or <vehicle>-<token>__anno-3DOD-EXT.json in older deliveries (same schema)
    od_files = sorted(glob.glob(join(sessions_root, session, 'annotation', 'anno-3DOD', '*.json')),
                      key=lambda p: os.path.basename(p) != 'anno-3DOD.json')
    od = json.load(open(od_files[0])) if od_files else None
    if od is None:
        print(f'[!] {session}: no anno-3DOD -> occlusion NOT checked', file=sys.stderr, flush=True)
    occ = None
    if od:
        T_e_cam = np.array(od['calibration']['CAM_FRONT']['T_parent_child'])
        T_e_lid = np.array(od['calibration']['LIDAR_TOP']['T_parent_child'])
        occ = Occlusion(cal, T_e_cam, map1, map2, *gt['image_size'])

    frames, n_kept, cls, other = [], 0, Counter(), Counter()
    for f in gt['frames']:
        for m in f['marks']:
            if m['layer'] == 'B3' and not is_target(m):
                other[m['name']] += 1
        if not f['image']:
            continue
        marks = [m for m in f['marks'] if is_target(m)]
        if not any(m['visible'] > 0 and m['distance_m'] <= max_dist for m in marks):
            continue  # cheap pre-filter: nothing in view even without occluders
        objs = (od['frames'].get(str(f['frame'])) or {}).get('objects', []) if od else []
        hidden = occ.frame_mask(objs, T_e_lid) if occ else None
        out_marks = []
        for m in marks:
            uv = cv2.fisheye.undistortPoints(np.array(m['pixels_raw'], np.float64).reshape(-1, 1, 2), K, D, P=new_K)
            uv = uv.reshape(-1, 2)
            rec = {k: m[k] for k in ('layer', 'id', 'name', 'kind', 'kind_code', 'distance_m', 'lateral_m')}
            rec['visible_map'] = m['visible']  # in frame & above bonnet (raw image), ignores occlusion
            if occ:
                frac, size = occ.visible(uv, m['layer'] == 'B3', hidden)
            else:
                frac, size = m['visible'], None
            rec['visible'] = round(frac, 2)
            rec['visible_px'] = size  # visible area [px^2] (polygons) / length [px] (stop lines)
            out_marks.append(rec | {'closed': m['layer'] == 'B3', 'pixels': np.round(uv, 1).tolist()})
        # usable mark: mostly visible; crossings / stop lines (repeating pattern, recognisable when cut) also when
        # a large part is visible. A sliver of an arrow is not recognisable -> arrows need the visible fraction.
        good = [m for m in out_marks if m['distance_m'] <= max_dist and
                (m['visible'] >= min_vis or (m['name'] in PARTIAL_OK and
                                             (m['visible_px'] or 0) >= (min_area if m['closed'] else min_len)))]
        if not good:
            continue
        n_kept += 1
        if (n_kept - 1) % every:
            continue
        cls.update(m['name'] for m in good)
        dst_img = join(img_dir, f['image'])
        if not isfile(dst_img):
            img = cv2.imread(join(src_dir, f['image']))
            if img is None:
                continue
            und = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)
            tmp = dst_img[:-4] + '.tmp.jpg'
            cv2.imwrite(tmp, und, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
            os.replace(tmp, dst_img)
        hidden_polys = []
        if hidden is not None and hidden.any():  # road hidden by 3D boxes, as polygons in undistorted px
            cs, _ = cv2.findContours(hidden.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            hidden_polys = [(c.reshape(-1, 2) * S + S // 2).tolist() for c in cs if len(c) >= 3]
        frames.append({'frame': f['frame'], 'timestamp': f['timestamp'], 'image': f'images/{session}/{f["image"]}',
                       'marks': out_marks, 'hidden': hidden_polys})

    keep = {os.path.basename(f['image']) for f in frames}
    for fn in os.listdir(img_dir):  # images from an earlier run with other settings
        if fn not in keep:
            os.remove(join(img_dir, fn))
    stats = {'frames_total': len(gt['frames']), 'frames_with_marks': n_kept, 'frames_written': len(frames),
             'occlusion_from_3dod': bool(od), 'class_counts': dict(cls), 'other_b3_kinds': dict(other)}
    lbl = {'session': session, 'image_size': gt['image_size'], 'K_undist': np.round(new_K, 4).tolist(),
           'selection': {'min_visible': min_vis, 'or_min_visible_area_px': min_area, 'or_min_visible_line_px': min_len,
                         'max_dist_m': max_dist, 'every': every,
                         'classes': sorted(TARGET_B3 | TARGET_B2)},
           'note': 'pixels = undistorted front image (fisheye, alpha=0). B3 = closed polygon, stop line = open polyline. '
                   'visible = fraction of the mark really seen (inside the image, above the bonnet, not behind an anno-3DOD box); '
                   'all target marks of a kept frame are listed so the converter can mark faint/far/hidden ones as ignore.',
           'stats': stats, 'frames': frames}
    json.dump(lbl, open(dst_lbl + '.tmp', 'w'), ensure_ascii=False)
    os.replace(dst_lbl + '.tmp', dst_lbl)
    return session, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--prepare', required=True, help='e2e_prepare export folder (session list in meta/header.json)')
    ap.add_argument('--sessions-root', required=True)
    ap.add_argument('--prep', required=True, help='folder with the vehicle *_camera_config.json / *_lidar_config.json')
    ap.add_argument('--out', required=True)
    ap.add_argument('--only', choices=['gt', 'build'])
    ap.add_argument('--limit', type=int)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--min-visible', type=float, default=0.5)
    ap.add_argument('--max-dist', type=float, default=30.0)
    ap.add_argument('--min-area', type=float, default=4000, help='crossings: ...or visible area >= this [px^2]')
    ap.add_argument('--min-len', type=float, default=200, help='stop lines: ...or visible length >= this [px]')
    ap.add_argument('--every', type=int, default=1, help='keep every Nth frame among the frames with marks (1 = all)')
    a = ap.parse_args()

    sessions = json.load(open(join(a.prepare, 'meta', 'header.json')))['scenario'][:a.limit]
    cam_cfg = glob.glob(join(a.prep, '*__camera_config.json'))
    assert len(cam_cfg) == 1, cam_cfg
    for d in ('gt', 'labels', 'images'):
        os.makedirs(join(a.out, d), exist_ok=True)
    t0 = time.time()

    if a.only != 'build':
        fails = []
        with ProcessPoolExecutor(a.workers) as ex:
            futs = [ex.submit(run_gt, s, a.sessions_root, cam_cfg[0], a.out) for s in sessions]
            for i, fu in enumerate(as_completed(futs), 1):
                s, st = fu.result()
                if st.startswith('FAIL'):
                    fails.append((s, st))
                print(f'[gt {i}/{len(sessions)}] {s} {st}', flush=True)
        if fails:
            print('GT failures:', *fails, sep='\n  ')
        sessions = [s for s in sessions if s not in dict(fails)]
    if a.only == 'gt':
        return

    total, per = Counter(), {}
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(run_build, s, a.sessions_root, a.prep, a.out, a.min_visible, a.max_dist, a.every,
                          a.min_area, a.min_len) for s in sessions]
        for i, fu in enumerate(as_completed(futs), 1):
            s, st = fu.result()
            per[s] = st
            print(f'[build {i}/{len(sessions)}] {s} kept {st["frames_written"]}/{st["frames_total"]} '
                  f'({time.time() - t0:.0f}s)', flush=True)
    for st in per.values():
        total['frames_total'] += st['frames_total']
        total['frames_written'] += st['frames_written']
    cls, other = Counter(), Counter()
    for st in per.values():
        cls.update(st['class_counts'])
        other.update(st['other_b3_kinds'])
    summary = {'sessions': len(per), **total, 'class_counts': dict(cls.most_common()),
               'other_b3_kinds_not_used': dict(other.most_common()), 'per_session': per}
    json.dump(summary, open(join(a.out, 'summary.json'), 'w'), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != 'per_session'}, ensure_ascii=False, indent=1))
    print(f'done in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
