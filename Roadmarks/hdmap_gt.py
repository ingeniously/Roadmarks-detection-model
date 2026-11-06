#!/usr/bin/env python3
"""Project the HD-map surface markings (B3: arrows, crosswalks) and line markings (B2: lane lines,
stop lines, U-turn zone lines ...) onto every front-camera frame -> per-frame ground truth.

Chain: map WGS84 -> ENU (session anchor) -> vehicle (rear axle on the ground, X fwd / Y left / Z up)
       -> camera (extrinsic) -> fisheye pixels (raw 1920x1536) -> ELAS pixels (640x480 resize)

usage: hdmap_gt.py <session folder> [--max-dist 40] [--draw out_dir] [--every 10] [--out file.json]
output: <session>/annotation/anno-HDMAP/hdmap_gt_front.json
"""
import argparse, glob, json, math, os, re
import numpy as np
import cv2

A_WGS, E2 = 6378137.0, 6.69437999014e-3  # WGS84
ELAS_W, ELAS_H = 640, 480


def bonnet_y(x):
    """top edge of the bonnet in the raw 1920x1536 front image (measured on this camera); below it nothing is visible"""
    return 1120 - 7.05e-5 * (np.asarray(x) - 960.0) ** 2


# HD-map kind name -> (English name, ELAS ROAD_SIGN id); 0 = no ELAS class
B3_KIND = {'직진': ('ahead', 1), '직진 및 우회전': ('ahead + right', 3), '직진 및 좌회전': ('ahead + left', 4),
           '우회전': ('right', 5), '좌회전': ('left', 6), '유턴': ('U-turn', 7), '좌회전 및 유턴': ('left + U-turn', 14),
           '직진 및 유턴': ('ahead + U-turn', 0), '좌우회전': ('left + right', 11),
           '횡단보도': ('crosswalk', 9), '고원식횡단보도': ('raised crosswalk', 9), '자전거횡단보도': ('bicycle crossing', 9),
           '오르막경사면': ('ramp (raised crosswalk slope)', 0)}
B2_KIND = {'차선': 'lane line', '진로변경제한선': 'no lane change line', '주정차금지선': 'no parking line', '중앙선': 'centre line',
           '유턴구역선': 'U-turn zone line', '정지선': 'stop line', '길가장자리구역선': 'road edge line', '유도선': 'guide line',
           '버스전용차선': 'bus lane line', '안전지대': 'safety zone'}


def b2_lmt(type_name, kind_name):
    """HD-map line type -> ELAS LMT code (0 = none / no ELAS equivalent)"""
    t = type_name or ''
    if kind_name == '유턴구역선': return 11                                        # SDB
    if kind_name == '정지선': return 0                                            # stop line is a road sign in ELAS (10)
    if t.startswith('백색-단선'): return 2 if t.endswith('점선') else 1           # SSB / SCB
    if t.startswith('황색-단선'): return 4 if t.endswith('점선') else 3           # SSA / SCA
    if t.startswith('황색-겹선'): return 5 if t.endswith('실선') else 6 if '좌점' in t else 7 if '우점' in t else 5
    if t.startswith('청색-단선'): return 9 if t.endswith('점선') else 8           # BSS / BSC
    if t.startswith('청색-겹선'): return 10 if t.endswith('실선') else 9          # BDC / (double dashed ~ BSS)
    return 0


def ecef(lat, lon, h):
    lat, lon = np.radians(lat), np.radians(lon)
    n = A_WGS / np.sqrt(1 - E2 * np.sin(lat) ** 2)
    return np.stack([(n + h) * np.cos(lat) * np.cos(lon), (n + h) * np.cos(lat) * np.sin(lon),
                     (n * (1 - E2) + h) * np.sin(lat)], -1)


class ENU:
    def __init__(self, lat0, lon0, h0):
        self.o = ecef(lat0, lon0, h0)
        la, lo = math.radians(lat0), math.radians(lon0)
        self.R = np.array([[-math.sin(lo), math.cos(lo), 0],
                           [-math.sin(la) * math.cos(lo), -math.sin(la) * math.sin(lo), math.cos(la)],
                           [math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)]])

    def __call__(self, lat, lon, h):
        return (ecef(np.asarray(lat), np.asarray(lon), np.asarray(h)) - self.o) @ self.R.T


def quat2R(w, x, y, z):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def veh2enu(heading_deg, pitch_deg, roll_deg):
    """vehicle FLU -> ENU. heading = true north, clockwise. pitch > 0 = nose up, roll > 0 = right side down."""
    yaw, p, r = math.radians(90 - heading_deg), math.radians(pitch_deg), math.radians(roll_deg)
    Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    Ry = np.array([[math.cos(p), 0, -math.sin(p)], [0, 1, 0], [math.sin(p), 0, math.cos(p)]])  # FLU: nose up = -rot about Y
    Rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
    return Rz @ Ry @ Rx


def densify(pts, step=0.25):
    out = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        n = max(1, int(np.linalg.norm(b - a) / step))
        out += [a + (b - a) * k / n for k in range(1, n + 1)]
    return np.array(out)


def simplify(uv, eps=1.5):
    """indices of the points kept by Ramer-Douglas-Peucker on the pixel polyline (max error eps px)"""
    keep = np.zeros(len(uv), bool)
    keep[[0, -1]] = True
    stack = [(0, len(uv) - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        d = uv[b] - uv[a]
        n = np.hypot(*d)
        seg = uv[a + 1:b] - uv[a]
        dist = np.abs(seg[:, 0] * d[1] - seg[:, 1] * d[0]) / n if n > 0 else np.hypot(seg[:, 0], seg[:, 1])
        i = int(np.argmax(dist))
        if dist[i] > eps:
            keep[a + 1 + i] = True
            stack += [(a, a + 1 + i), (a + 1 + i, b)]
    return np.nonzero(keep)[0]


def clip_near(pts, near, closed):
    """keep the part of a polyline / polygon (camera frame) with z >= near; returns a list of pieces"""
    if closed:  # Sutherland-Hodgman against the plane z = near
        out = []
        for i in range(len(pts)):
            a, b = pts[i - 1], pts[i]
            ina, inb = a[2] >= near, b[2] >= near
            if ina != inb:
                out.append(a + (b - a) * (near - a[2]) / (b[2] - a[2]))
            if inb:
                out.append(b)
        return [np.array(out)] if len(out) >= 3 else []
    pieces, cur = [], []
    for i in range(len(pts)):
        if pts[i][2] >= near:
            if i > 0 and pts[i - 1][2] < near:
                a, b = pts[i - 1], pts[i]
                cur.append(a + (b - a) * (near - a[2]) / (b[2] - a[2]))
            cur.append(pts[i])
        else:
            if cur:
                a, b = pts[i - 1], pts[i]
                cur.append(a + (b - a) * (near - a[2]) / (b[2] - a[2]))
                pieces.append(np.array(cur))
            cur = []
    if cur:
        pieces.append(np.array(cur))
    return [p for p in pieces if len(p) >= 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('session')
    ap.add_argument('--max-dist', type=float, default=40.0, help='ignore map features farther than this [m]')
    ap.add_argument('--near', type=float, default=1.0, help='clip geometry closer than this to the camera [m]')
    ap.add_argument('--draw', help='folder for overlay images (raw + ELAS size)')
    ap.add_argument('--out', help='output json (default: <session>/annotation/anno-HDMAP/hdmap_gt_front.json)')
    ap.add_argument('--every', type=int, default=1, help='draw only every Nth frame')
    ap.add_argument('--cam-config', help='camera_config.json (default: <session>/*camera_config.json)')
    args = ap.parse_args()
    S = args.session.rstrip('/')

    hd = json.load(open(glob.glob(S + '/annotation/anno-HDMAP/hdmap.json')[0]))
    pose_files = (glob.glob(S + '/annotation/anno-LOCAL/*pose_REFINED.json') or
                  glob.glob(S + '/annotation/anno-POSE_REFINE/*pose_REFINED.json'))  # server layout
    pose = json.load(open(pose_files[0]))
    cams = json.load(open(args.cam_config or glob.glob(S + '/*camera_config.json')[0]))
    cam = [c for c in cams if c['Position'] == 'CAM_FRONT'][0]

    co = pose['header']['coordinate']
    enu = ENU(*co['anchor_wgs84_ellip'])
    lever = np.array(co['lever_arm_m'])  # GNSS antenna in the vehicle frame (rear axle on the ground, FLU)

    ex, it = cam['extrinsic'], cam['intrinsic']
    R_c2v = quat2R(ex['w'], ex['x'], ex['y'], ex['z'])
    t_c2v = np.array([ex['tx'], ex['ty'], ex['tz']])
    K = np.array([[it['fx'], it['skew'], it['cx']], [0, it['fy'], it['cy']], [0, 0, 1]])
    D = np.array([it['k1'], it['k2'], it['k3'], it['k4']])
    W, H = cam['image_size_x'], cam['image_size_y']
    sx, sy = ELAS_W / W, ELAS_H / H

    # map features in ENU (map heights are ellipsoidal, like the anchor)
    feats = []
    for layer, closed in (('B3_SURFACEMARK', True), ('B2_SURFACELINEMARK', False)):
        for f in hd['layers'].get(layer, []):  # a route without any surface mark has no B3 layer
            if f['geometry_type'] == 'Point':
                continue
            g = np.array(f['wgs84'], float)
            p = enu(g[:, 0], g[:, 1], g[:, 2])
            if closed and np.allclose(p[0], p[-1]):
                p = p[:-1]
            e = f['attributes']['expanded']
            feats.append({'layer': layer[:2], 'id': f['ngii_id'], 'type': e.get('type_name'), 'kind': e.get('kind_name'),
                          'kind_code': e.get('kind'), 'closed': closed, 'enu': p, 'c': p.mean(0)})

    imgs = sorted(glob.glob(S + '/perception_sensor_data/cam-rgb-front/*.jpg'))
    by_t = {int(re.search(r'__t(\d+)', f).group(1)): f for f in imgs}
    if args.draw:
        os.makedirs(args.draw, exist_ok=True)

    out = {'source': 'HD map B2/B3 projected with pose_REFINED + CAM_FRONT calibration',
           'image_size': [W, H], 'elas_size': [ELAS_W, ELAS_H], 'frames': []}
    for k in sorted(pose['frames'], key=int):
        fr = pose['frames'][k]
        d, t = fr['data'], fr['timestamp']
        rec = {'frame': int(k), 'elas_image': 'lane_%d.png' % int(k), 'timestamp': t, 'image': os.path.basename(by_t.get(t, '')) or None, 'marks': []}
        if d.get('lat') is None:
            out['frames'].append(rec)
            continue
        R_v2e = veh2enu(d['heading_deg'], d['pitch_deg'] or 0.0, d['roll_deg'] or 0.0)
        ant = enu(d['lat'], d['lon'], d['alt'] + (d.get('geoid_undulation_m') or co['geoid_N_m']))
        ego = ant - R_v2e @ lever
        for f in feats:
            if np.linalg.norm(f['c'][:2] - ego[:2]) > args.max_dist + 30:
                continue
            pv = (f['enu'] - ego) @ R_v2e                  # ENU -> vehicle
            if f['closed']:
                pv = np.vstack([pv, pv[:1]])
            pv = densify(pv)[:-1] if f['closed'] else densify(pv)
            pv = pv[np.hypot(pv[:, 0], pv[:, 1]) <= args.max_dist + 5] if not f['closed'] else pv
            if len(pv) < 2:
                continue
            pc = (pv - t_c2v) @ R_c2v                       # vehicle -> camera
            for piece in clip_near(pc, args.near, f['closed']):
                if f['closed'] and np.hypot(*((piece @ R_c2v.T + t_c2v)[:, :2].mean(0))) > args.max_dist:
                    continue
                uv, _ = cv2.fisheye.projectPoints(piece.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, D)
                uv = uv.reshape(-1, 2)
                # visible = inside the image and above the bonnet (area for polygons, points for lines)
                if f['closed']:
                    mask = np.zeros((H // 4, W // 4), np.uint8)
                    cv2.fillPoly(mask, [np.round(uv / 4).astype(np.int32)], 1)
                    total = int(mask.sum())
                    yy, xx = np.nonzero(mask)
                    seen = int((yy * 4 < bonnet_y(xx * 4)).sum())
                    visible = seen / total if total else 0.0
                else:
                    visible = float(((uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < bonnet_y(uv[:, 0]))).mean())
                if visible <= 0:
                    continue
                pv_piece = piece @ R_c2v.T + t_c2v
                dist_m = round(float(np.hypot(*pv_piece[:, :2].mean(0))), 2)
                lat_m = round(float(pv_piece[:, 1].mean()), 2)
                keep = simplify(uv)
                uv, pv_piece = uv[keep], pv_piece[keep]
                if f['layer'] == 'B3':
                    name, elas = B3_KIND.get(f['kind'], (f['kind'], 0))
                    extra = {'name': name, 'elas_road_sign': elas}
                else:
                    extra = {'name': B2_KIND.get(f['kind'], f['kind']), 'elas_lmt': b2_lmt(f['type'], f['kind']),
                             'elas_road_sign': 10 if f['kind'] == '정지선' else 0}
                rec['marks'].append({
                    'layer': f['layer'], 'id': f['id'], **extra, 'type': f['type'], 'kind': f['kind'], 'kind_code': f['kind_code'],
                    'visible': round(visible, 2),
                    'distance_m': dist_m,
                    'lateral_m': lat_m,       # > 0 = left of the car
                    'vehicle_xy_m': np.round(pv_piece[:, :2], 2).tolist(),
                    'pixels_raw': np.round(uv, 1).tolist(),
                    'pixels_elas': np.round(uv * [sx, sy], 1).tolist()})
        out['frames'].append(rec)

        if args.draw and rec['image'] and int(k) % args.every == 0:
            img = cv2.imread(by_t[t])
            for m in rec['marks']:
                pts = np.round(np.array(m['pixels_raw'])).astype(np.int32)
                if m['layer'] == 'B3':
                    col = (0, 0, 255) if '횡단' in (m['type'] or '') else (255, 0, 255)
                    cv2.polylines(img, [pts], True, col, 3)
                else:
                    col = (0, 215, 255) if '황색' in (m['type'] or '') else (255, 140, 0) if '청색' in (m['type'] or '') else (0, 255, 0)
                    cv2.polylines(img, [pts], False, col, 2)
            xs = np.arange(0, W + 1, 20)
            cv2.polylines(img, [np.stack([xs, bonnet_y(xs)], 1).astype(np.int32)], False, (0, 0, 255), 1)
            for m in rec['marks']:
                if m['layer'] == 'B3' and m['visible'] >= 0.5:
                    p = np.array(m['pixels_raw']).min(0).astype(int)
                    cv2.putText(img, m['name'], (int(p[0]), max(20, int(p[1]) - 8)), 0, 1.0, (255, 0, 255), 2)
            cv2.putText(img, 'frame %s' % k, (30, 60), 0, 1.5, (0, 255, 255), 3)
            cv2.imwrite(os.path.join(args.draw, 'gt_%04d.jpg' % int(k)), cv2.resize(img, (960, 768)))

    dst = args.out or S + '/annotation/anno-HDMAP/hdmap_gt_front.json'
    json.dump(out, open(dst, 'w'), ensure_ascii=False)
    n = sum(1 for f in out['frames'] for m in f['marks'] if m['layer'] == 'B3')
    print('wrote', dst, '-', len(out['frames']), 'frames,', n, 'surface-mark projections')


if __name__ == '__main__':
    main()