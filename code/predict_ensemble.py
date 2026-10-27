#!/usr/bin/env python3
"""Multi-model / multi-scale / flip-TTA prediction with mask-level dedup."""
import argparse, csv
from pathlib import Path
import numpy as np, cv2
from ultralytics import YOLO
from pycocotools import mask as mu

BASE = Path('/data/research/sota_method_2026-10-24/solar_filament')
TEST = BASE / 'data/raw/MAGFiLO_1.0_Kaggle_2026/test/test_images'


def iou(a, b):
    inter = np.logical_and(a, b).sum()
    return 0.0 if inter == 0 else inter / float(a.sum() + b.sum() - inter)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', nargs='+', required=True)
    ap.add_argument('--imgsz', nargs='+', type=int, default=[1280])
    ap.add_argument('--conf', type=float, default=0.3)
    ap.add_argument('--flip', action='store_true')
    ap.add_argument('--dup-iou', type=float, default=0.8)
    ap.add_argument('--out', default=str(BASE / 'out/ensemble.csv'))
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--test-dir', default=str(TEST))
    a = ap.parse_args()
    files = sorted(Path(a.test_dir).glob('*.jpeg')) + sorted(Path(a.test_dir).glob('*.jpg'))
    if a.limit:
        files = files[:a.limit]
    models = [YOLO(w) for w in a.weights]
    rows = []; empty = 0
    for k, f in enumerate(files):
        src = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        H, W = src.shape
        cands = []

        def add(masks, confs, flip):
            for j in range(masks.shape[0]):
                mm = (masks[j] > 0.5).astype(np.uint8)
                if flip == 1:
                    mm = mm[:, ::-1]
                elif flip == 2:
                    mm = mm[::-1]
                elif flip == 3:
                    mm = mm[::-1, ::-1]
                if mm.sum() == 0:
                    continue
                cands.append((float(confs[j]), np.ascontiguousarray(mm)))

        for m in models:
            for s in a.imgsz:
                r = m.predict(str(f), imgsz=s, conf=a.conf, iou=0.7, max_det=100,
                              retina_masks=True, verbose=False)[0]
                if r.masks is not None:
                    add(r.masks.data.cpu().numpy(), r.boxes.conf.cpu().numpy(), 0)
        if a.flip:
            for m in models:
                for s in a.imgsz:
                    for fl, img in ((1, src[:, ::-1]), (2, src[::-1]),
                                    (3, src[::-1, ::-1])):
                        r = m.predict(np.ascontiguousarray(img), imgsz=s, conf=a.conf, iou=0.7,
                                      max_det=100, retina_masks=True, verbose=False)[0]
                        if r.masks is not None:
                            # undo the flip on the original-size mask
                            add(r.masks.data.cpu().numpy(), r.boxes.conf.cpu().numpy(),
                                {1: 1, 2: 2, 3: 3}[fl])
        cands.sort(key=lambda x: -x[0])
        kept = []
        for c, mm in cands:
            if all(iou(mm, km) < a.dup_iou for _, km in kept):
                kept.append((c, mm))
        if not kept:
            empty += 1
        for i, (c, mm) in enumerate(kept):
            rows.append({'filament_id': f'{f.stem}_{i}',
                         'segmentation_rle': mu.encode(np.asfortranarray(mm))['counts'].decode()})
        if (k + 1) % 20 == 0:
            print(f'{k+1}/{len(files)} rows={len(rows)} empty={empty}', flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['filament_id', 'segmentation_rle'])
        w.writeheader(); w.writerows(rows)
    print(f'wrote {a.out}: {len(rows)} rows, {empty} empty images')


if __name__ == '__main__':
    main()
