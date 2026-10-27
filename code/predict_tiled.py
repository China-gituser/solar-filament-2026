#!/usr/bin/env python3
"""Tiled inference: run the detector on overlapping crops so that small
filaments occupy more pixels, then merge masks back in image coordinates.

Full-image inference at 1280 shrinks a 2048 image by 0.625; a 1024 crop is fed
at ~1.25x, which is where small targets stop being sub-pixel.
"""
import argparse, json, sys, collections
from pathlib import Path
import cv2, numpy as np
from ultralytics import YOLO
from pycocotools import mask as mu

sys.path.insert(0, '/data/research/sota_method_2026-10-24/solar_filament/code')
from pq_official import evaluate
from eval_yolo_val import build_gt

BASE = Path('/data/research/sota_method_2026-10-24/solar_filament')
RAW = BASE / 'data/raw/MAGFiLO_1.0_Kaggle_2026'


def tiles(h, w, size, stride):
    ys = list(range(0, max(h - size, 0) + 1, stride))
    xs = list(range(0, max(w - size, 0) + 1, stride))
    if ys[-1] + size < h: ys.append(h - size)
    if xs[-1] + size < w: xs.append(w - size)
    return [(y, x) for y in ys for x in xs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--n', type=int, default=50)
    ap.add_argument('--tile', type=int, default=1024)
    ap.add_argument('--stride', type=int, default=768)
    ap.add_argument('--imgsz', type=int, default=1024)
    ap.add_argument('--conf', type=float, default=0.3)
    ap.add_argument('--dup-iou', type=float, default=0.6)
    a = ap.parse_args()
    split = json.load(open(BASE / 'data/split.json'))['val'][:a.n]
    gt = build_gt(split)
    m = YOLO(a.weights)
    pm = {}
    for fi, f in enumerate(split):
        gray = cv2.imread(str(RAW / 'train/train_images' / f), cv2.IMREAD_GRAYSCALE)
        H, W = gray.shape
        cands = []
        for (y, x) in tiles(H, W, a.tile, a.stride):
            crop = np.ascontiguousarray(gray[y:y+a.tile, x:x+a.tile])
            r = m.predict(crop, imgsz=a.imgsz, conf=a.conf, iou=0.7, max_det=100,
                          retina_masks=True, verbose=False)[0]
            if r.masks is None: continue
            ms = r.masks.data.cpu().numpy(); cf = r.boxes.conf.cpu().numpy()
            for j in range(ms.shape[0]):
                full = np.zeros((H, W), np.uint8)
                mm = (ms[j] > 0.5).astype(np.uint8)
                full[y:y+a.tile, x:x+a.tile] = mm
                if full.sum() == 0: continue
                cands.append((full, float(cf[j])))
        if not cands:
            pm[Path(f).stem] = []; continue
        cands.sort(key=lambda z: -z[1])
        kept = []
        for mm, c in cands:
            dup = False
            for km, _ in kept:
                inter = np.logical_and(mm, km).sum()
                if inter / max(mm.sum() + km.sum() - inter, 1) > a.dup_iou:
                    dup = True; break
            if not dup: kept.append((mm, c))
        occ = None; out = []
        for mm, c in kept:
            if occ is not None:
                mm = np.logical_and(mm, np.logical_not(occ)).astype(np.uint8)
            if mm.sum() == 0: continue
            occ = mm if occ is None else np.logical_or(occ, mm)
            out.append(mm)
        pm[Path(f).stem] = [mu.encode(np.asfortranarray(z))['counts'].decode() for z in out]
        if (fi + 1) % 20 == 0: print(f'  {fi+1}/{len(split)}', flush=True)
    r = evaluate(gt, pm)
    print(json.dumps(r, indent=1))


if __name__ == '__main__':
    main()
