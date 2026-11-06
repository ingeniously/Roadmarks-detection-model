#!/usr/bin/env python
"""Undistort whole sessions (all 6 cameras) with MotifE2E_Demo's s3_undistort, using the calibration in --prep.

Writes <data-dir>/<session>/perception_sensor_data_undist/ next to perception_sensor_data/, so <data-dir> must be a
LOCAL, writable copy of the sessions (never run it on the shared server folder).
NOT needed for training: build_roadmark_dataset.py undistorts the front frames it keeps by itself, identically.

usage: python Roadmarks/run_undistort.py --data-dir <local sessions folder> --sessions 20260910_070626 ... [--workers 8]
"""
import argparse, multiprocessing, sys
from os.path import abspath, dirname, join

WS = abspath(join(dirname(abspath(__file__)), '..'))  # workspace


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-dir', required=True, help='local folder that contains the session folders')
    ap.add_argument('--sessions', nargs='+', required=True)
    ap.add_argument('--prep', default=join(WS, 'prep'), help='folder with *__camera_config.json')
    ap.add_argument('--repo', default=join(WS, 'MotifE2E_Demo'))
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()

    multiprocessing.set_start_method('fork')  # workers inherit the calibration-path patch below
    sys.path.insert(0, a.repo)
    import tools.data_pipeline.common.layout as L
    L.config_search_roots = lambda d: [abspath(a.prep)]  # the sessions themselves carry no camera_config
    sys.path.insert(0, join(a.repo, 'tools', 'data_pipeline', 's3_undistort'))
    import undistort
    undistort.run(a.sessions, a.workers, abspath(a.data_dir), 0.0, False)  # alpha 0, keep the originals


if __name__ == '__main__':
    main()
