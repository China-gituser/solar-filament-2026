#!/usr/bin/env python3
"""Refine YOLO proposals with SAM2 box prompts.

The error analysis shows misses sit at IoU ~0.27: the detector finds the
filament but the mask boundary is off. SAM2 conditions on the box and predicts
a boundary from the image itself, which is exactly the missing piece.
"""
import argparse, json, sys, collections
from pathlib import Path
import cv2, numpy as np
from ultralytics import YOLO, SAM
from pycocotools import mask as mu

sys.path.insert(0, '/data/research/sota_method_2026-10-24/solar_filament/code')
from pq_official import evaluate
from eval_yolo_val import build_gt

BASE = Path('/data/research/sota_method_2026-10-24/solar_filament')
RAW = BASE / 'data/raw/MAGFiLO_1.0_Kaggle_2026'
SAMW = '/data/research/sota_method_2026-10-24/geoai/sam2.1_b.pt'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--yolo', default=str(BASE / 'runs/s1280/weights/best.pt'))
    ap.add_argument('--sam', default=SAMW)
    ap.add_argument('--n', type=int, default=30)
    ap.add_argument('--conf', type=float, default=0.5)
    ap.add_argument('--min-area', type=int, default=200)
    a = ap.parse_args()
    split = json.load(open(BASE / 'data/split.json'))['val'][:a.n]
    gt = build_gt(split)
    ym = YOLO(a.yolo); sm = SAM(a.sam)
    pm = {}
    for k, f in enumerate(split):
        path = str(RAW / 'train/train_images' / f)
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        r = ym.predict(path, imgsz=1280, conf=a.conf, iou=0.7, max_det=100,
                       retina_masks=True, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            pm[Path(f).stem] = []; continue
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        try:
            sres = sm(rgb, bboxes=boxes, verbose=False)[0]
            masks = sres.masks.data.cpu().numpy() if sres.masks is not None else None
        except Exception as e:
            print('sam failed', f, e, flush=True); masks = None
        if masks is None or len(masks) != len(boxes):
            masks = r.masks.data.cpu().numpy() if r.masks is not None else None
        items = []
        if masks is not None:
            n = min(len(masks), len(boxes))
            for j in range(n):
                m = (masks[j] > 0.5).astype(np.uint8)
                if m.shape != gray.shape:
                    m = cv2.resize(m, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
                if m.sum() < a.min_area:
                    m = (r.masks.data.cpu().numpy()[j] > 0.5).astype(np.uint8) if r.masks is not None else m
                if m.sum() == 0: continue
                items.append((m, float(confs[j])))
        items.sort(key=lambda z: -z[1])
        occ = None; out = []
        for m, c in items:
            if occ is not None:
                m = np.logical_and(m, np.logical_not(occ)).astype(np.uint8)
            if m.sum() == 0: continue
            occ = m if occ is None else np.logical_or(occ, m)
            out.append(m)
        pm[Path(f).stem] = [mu.encode(np.asfortranarray(z))['counts'].decode() for z in out]
        if (k + 1) % 10 == 0: print(f'  {k+1}/{len(split)}', flush=True)
    res = evaluate(gt, pm)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
