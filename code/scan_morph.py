#!/usr/bin/env python3
"""Is the predicted mask too large? Erode/dilate by k pixels and rescore."""
import argparse, json, sys
from pathlib import Path
import numpy as np, cv2
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
    a = ap.parse_args()
    split = json.load(open(BASE / 'data/split.json'))['val'][:a.n]
    gt = build_gt(split)
    # GT area reference
    from pq_official import decode
    ga = [int(decode(v).sum()) for v in gt.values() if v]
    print('GT  instances=%d mean_area=%.0f median_area=%.0f' % (
        len(ga), np.mean(ga), np.median(ga)), flush=True)
    model = YOLO(a.weights)
    raw = {}
    for k, f in enumerate(split):
        r = model.predict(str(RAW / 'train/train_images' / f), imgsz=a.imgsz, conf=a.conf,
                          iou=0.7, max_det=100, retina_masks=True, verbose=False)[0]
        raw[Path(f).stem] = [] if r.masks is None else [
            (r.masks.data.cpu().numpy()[j] > 0.5).astype(np.uint8)
            for j in range(r.masks.data.shape[0])]
    for k in (-4, -3, -2, -1, 0, 1, 2, 3, 4):
        if k == 0:
            op = lambda m: m
        else:
            ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (abs(k) * 2 + 1,) * 2)
            op = (lambda ker: (lambda m: cv2.erode(m, ker)))(ker) if k < 0 else \
                 (lambda ker: (lambda m: cv2.dilate(m, ker)))(ker)
        pm = {}; areas = []
        for stem, masks in raw.items():
            rles = []
            for m in masks:
                mm = op(m)
                if mm.sum() == 0:
                    continue
                areas.append(int(mm.sum()))
                rles.append(mu.encode(np.asfortranarray(mm))['counts'].decode())
            pm[stem] = rles
        res = evaluate(gt, pm)
        print(f'{"erode" if k<0 else "dilate"} {abs(k)}px: PQ={res["pq"]:.4f} SQ={res["sq"]:.4f} '
              f'RQ={res["rq"]:.4f} TP={res["tp"]} FP={res["fp"]} FN={res["fn"]} '
              f'mean_area={np.mean(areas) if areas else 0:.0f}', flush=True)


if __name__ == '__main__':
    main()
