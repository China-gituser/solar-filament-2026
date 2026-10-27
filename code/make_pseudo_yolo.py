#!/usr/bin/env python3
"""Convert a submission CSV into YOLO polygon labels on the test images."""
import argparse, csv, collections, os
from pathlib import Path
import cv2, numpy as np
from pycocotools import mask as mu

BASE = Path('/data/research/sota_method_2026-10-24/solar_filament')
TEST = BASE / 'data/raw/MAGFiLO_1.0_Kaggle_2026/test/test_images'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default=str(BASE / 'out/sub_refine_s1280.csv'))
    ap.add_argument('--out', default=str(BASE / 'yolo_pseudo'))
    ap.add_argument('--min-area', type=int, default=150)
    a = ap.parse_args()
    rows = collections.defaultdict(list)
    for r in csv.DictReader(open(a.csv)):
        fid = r['filament_id']; stem = fid.split('_', 1)[0]
        rows[stem].append(r['segmentation_rle'])
    out = Path(a.out)
    (out / 'images/train').mkdir(parents=True, exist_ok=True)
    (out / 'labels/train').mkdir(parents=True, exist_ok=True)
    n_inst = 0
    for f in sorted(TEST.glob('*.jpeg')):
        stem = f.stem
        dst = out / 'images/train' / f.name
        if not dst.exists():
            os.symlink(f, dst)
        lines = []
        for rle in rows.get(stem, []):
            m = mu.decode({'size': [2048, 2048], 'counts': rle.encode()}).astype(np.uint8)
            if m.sum() < a.min_area:
                continue
            cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cs:
                continue
            c = max(cs, key=cv2.contourArea)
            ap2 = cv2.approxPolyDP(c, 1.5, True).reshape(-1, 2).astype(float)
            if len(ap2) < 3:
                continue
            ap2[:, 0] /= 2048.0; ap2[:, 1] /= 2048.0
            lines.append('0 ' + ' '.join(f'{x:.6f} {y:.6f}' for x, y in ap2))
            n_inst += 1
        (out / 'labels/train' / f'{stem}.txt').write_text('\n'.join(lines) + ('\n' if lines else ''))
    print('pseudo images', len(list((out/'labels/train').glob('*.txt'))), 'instances', n_inst)


if __name__ == '__main__':
    main()
