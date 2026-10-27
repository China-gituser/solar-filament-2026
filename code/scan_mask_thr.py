#!/usr/bin/env python3
"""Does the predicted mask need shrinking? Sweep the mask binarisation threshold."""
import argparse, json, sys
from pathlib import Path
import numpy as np
from ultralytics import YOLO
from pycocotools import mask as mu

sys.path.insert(0, '/data/research/sota_method_2026-10-24/solar_filament/code')
from pq_official import evaluate
from eval_yolo_val import build_gt

BASE = Path('/data/research/sota_method_2026-10-24/solar_filament')
RAW = BASE / 'data/raw/MAGFiLO_1.0_Kaggle_2026'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--imgsz', type=int, default=1280)
    ap.add_argument('--n', type=int, default=25)
    ap.add_argument('--conf', type=float, default=0.3)
    ap.add_argument('--thrs', default='0.3,0.4,0.5,0.6,0.7,0.8,0.9')
    a = ap.parse_args()
    split = json.load(open(BASE / 'data/split.json'))['val'][:a.n]
    gt = build_gt(split)
    model = YOLO(a.weights)
    raw = {}
    for k, f in enumerate(split):
        r = model.predict(str(RAW / 'train/train_images' / f), imgsz=a.imgsz, conf=a.conf,
                          iou=0.7, max_det=100, retina_masks=True, verbose=False)[0]
        items = []
        if r.masks is not None:
            masks = r.masks.data.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            items = [(float(confs[j]), masks[j]) for j in range(masks.shape[0])]
        raw[Path(f).stem] = items
        if (k + 1) % 20 == 0:
            print(f'  {k+1}/{len(split)}', flush=True)
    print('mean predicted area / GT-ish check by threshold:')
    for t in [float(x) for x in a.thrs.split(',')]:
        pm = {}
        areas = []
        for stem, items in raw.items():
            rles = []
            for c, m in items:
                mm = (m > t).astype(np.uint8)
                if mm.sum() == 0:
                    continue
                areas.append(int(mm.sum()))
                rles.append(mu.encode(np.asfortranarray(mm))['counts'].decode())
            pm[stem] = rles
        res = evaluate(gt, pm)
        print(f'mask_thr={t:.2f}: PQ={res["pq"]:.4f} SQ={res["sq"]:.4f} RQ={res["rq"]:.4f} '
              f'TP={res["tp"]} FP={res["fp"]} FN={res["fn"]} mean_area={np.mean(areas) if areas else 0:.0f}')


if __name__ == '__main__':
    main()
