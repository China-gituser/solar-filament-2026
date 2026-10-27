#!/usr/bin/env python3
"""Predict the Kaggle test split and write the submission CSV.

Format: filament_id,segmentation_rle
  filament_id = <physical image stem>_<index>
  segmentation_rle = compressed COCO RLE, ASCII counts
"""
import argparse, csv, collections
from pathlib import Path

import numpy as np
from ultralytics import YOLO
from pycocotools import mask as mu

BASE = Path('/data/research/sota_method_2026-10-24/solar_filament')
TEST = BASE / 'data/raw/MAGFiLO_1.0_Kaggle_2026/test/test_images'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', nargs='+', required=True)
    ap.add_argument('--imgsz', type=int, default=1280)
    ap.add_argument('--conf', type=float, default=0.05)
    ap.add_argument('--iou', type=float, default=0.7)
    ap.add_argument('--max-det', type=int, default=50)
    ap.add_argument('--out', default=str(BASE / 'out/submission.csv'))
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()

    files = sorted(TEST.glob('*.jpeg')) + sorted(TEST.glob('*.jpg'))
    if a.limit:
        files = files[:a.limit]
    models = [YOLO(w) for w in a.weights]
    rows = []
    empty = 0
    for k, f in enumerate(files):
        cands = []   # (mask, score)
        for m in models:
            for s in (a.imgsz,) if isinstance(a.imgsz, int) else a.imgsz:
                r = m.predict(str(f), imgsz=s, conf=a.conf, iou=a.iou, max_det=a.max_det,
                              retina_masks=True, verbose=False)[0]
                if r.masks is None:
                    continue
                masks = r.masks.data.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                for j in range(masks.shape[0]):
                    mm = (masks[j] > 0.5).astype(np.uint8)
                    if mm.sum() == 0:
                        continue
                    cands.append((mm, float(confs[j])))
        # Kaggle rejects submissions with overlapping masks: keep the most
        # confident prediction for every pixel and drop what is left of the rest
        cands.sort(key=lambda x: -x[1])
        kept = []
        occupied = np.zeros_like(cands[0][0]) if cands else None
        for mm, sc in cands:
            mm = np.logical_and(mm, np.logical_not(occupied)).astype(np.uint8)
            if mm.sum() == 0:
                continue
            occupied = np.logical_or(occupied, mm)
            kept.append(mm)
        if not kept:
            empty += 1
        for i, mm in enumerate(kept):
            rle = mu.encode(np.asfortranarray(mm))
            rows.append({'filament_id': f'{f.stem}_{i}',
                         'segmentation_rle': rle['counts'].decode('ascii')})
        if (k + 1) % 20 == 0:
            print(f'{k+1}/{len(files)} rows={len(rows)} empty={empty}', flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['filament_id', 'segmentation_rle'])
        w.writeheader()
        w.writerows(rows)
    print(f'wrote {a.out}: {len(rows)} rows over {len(files)} images '
          f'({empty} images with no prediction)')


if __name__ == '__main__':
    main()
