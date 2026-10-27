#!/usr/bin/env python3
"""One low-threshold inference pass, then score official PQ at several conf cutoffs."""
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
    ap.add_argument('--n', type=int, default=40)
    ap.add_argument('--conf', type=float, default=0.001)
    ap.add_argument('--thrs', default='0.001,0.01,0.02,0.05,0.1,0.15,0.2,0.3,0.4,0.5')
    a = ap.parse_args()

    split = json.load(open(BASE / 'data/split.json'))['val'][:a.n]
    gt = build_gt(split)
    model = YOLO(a.weights)
    preds = {}
    for k, f in enumerate(split):
        r = model.predict(str(RAW / 'train/train_images' / f), imgsz=a.imgsz,
                          conf=a.conf, iou=0.7, max_det=100, retina_masks=True,
                          verbose=False)[0]
        items = []
        if r.masks is not None:
            masks = r.masks.data.cpu().numpy()
            conf = r.boxes.conf.cpu().numpy()
            cands = [((masks[j] > 0.5).astype(np.uint8), float(conf[j]))
                     for j in range(masks.shape[0])]
            cands.sort(key=lambda x: -x[1])
            occupied = None
            for mm, c in cands:
                if mm.sum() == 0:
                    continue
                if occupied is not None:
                    mm = np.logical_and(mm, np.logical_not(occupied)).astype(np.uint8)
                if mm.sum() == 0:
                    continue
                occupied = mm if occupied is None else np.logical_or(occupied, mm)
                items.append((c, mu.encode(np.asfortranarray(mm))['counts'].decode()))
        preds[Path(f).stem] = items
        if (k + 1) % 20 == 0:
            print(f'  {k+1}/{len(split)}', flush=True)
    for thr in [float(x) for x in a.thrs.split(',')]:
        pm = {k: [r for c, r in v if c >= thr] for k, v in preds.items()}
        res = evaluate(gt, pm)
        print(f'conf>={thr:.3f}: PQ={res["pq"]:.4f} SQ={res["sq"]:.4f} RQ={res["rq"]:.4f} '
              f'TP={res["tp"]} FP={res["fp"]} FN={res["fn"]} n_pred={sum(len(v) for v in pm.values())}')


if __name__ == '__main__':
    main()
