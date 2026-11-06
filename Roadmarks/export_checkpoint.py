#!/usr/bin/env python
"""Turn a training checkpoint into a small release checkpoint: model weights (the EMA weights, which mmengine's
EMAHook already stores in `state_dict`) + class names. Optimizer state, EMA copy and logs are dropped.

usage: python Roadmarks/export_checkpoint.py work_dirs/rtmdet-ins_s_roadmark/best_coco_segm_mAP_epoch_N.pth \
                                             rtmdet-ins_s_roadmark_v1.pth
"""
import argparse, hashlib, os

import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('src')
    ap.add_argument('dst')
    a = ap.parse_args()
    c = torch.load(a.src, map_location='cpu', weights_only=False)  # our own training checkpoint
    m = c.get('meta', {})
    out = {'state_dict': c['state_dict'],
           'meta': {'dataset_meta': {'classes': list(m.get('dataset_meta', {}).get('classes', []))},
                    'epoch': m.get('epoch'), 'source': os.path.basename(a.src)}}
    torch.save(out, a.dst)
    sha = hashlib.sha256(open(a.dst, 'rb').read()).hexdigest()[:8]
    n = sum(v.numel() for v in out['state_dict'].values() if v.is_floating_point())
    print(f'{a.dst}: {os.path.getsize(a.dst) / 1e6:.1f} MB, {n / 1e6:.1f} M params, '
          f'epoch {out["meta"]["epoch"]}, sha256 {sha}')


if __name__ == '__main__':
    main()
