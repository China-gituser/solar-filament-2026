#!/usr/bin/env python3
"""Predict the held-out validation images with a YOLO-seg checkpoint and score
them with the organisers' PQ (per annotator pass, pooled)."""
import argparse, json, collections
from pathlib import Path

import numpy as np
from ultralytics import YOLO
from pycocotools import mask as mu

import sys
sys.path.insert(0, '/data/research/sota_method_2026-10-24/solar_filament/code')
from pq_official import evaluate, rasterize, H, W

BASE = Path('/data/research/sota_method_2026-10-24/solar_filament')
RAW = BASE / 'data/raw/MAGFiLO_1.0_Kaggle_2026'


def build_gt(val_files):
    """Load cached RLEs for every annotator pass of the requested physical images."""
    d = json.load(open(RAW / 'train/MAGFiLO_1.0_Annotations_kaggle2026_train.json'))
    imgs = {str(i['id']): i for i in d['images']}
    cache = json.load(open(BASE / 'data/gt_rle_train.json'))
    want = set(val_files)
    gt = {}
    for iid, im in imgs.items():
        if im['file_name'] in want:
            gt[str(iid)] = cache.get(str(iid), [])
    return gt


def predict_map(model, files, imgsz, conf, iou, max_det, retina):
    out = collections.defaultdict(list)
    for k, f in enumerate(files):
        img = str(RAW / 'train/train_images' / f)
        r = model.predict(img, imgsz=imgsz, conf=conf, iou=iou, max_det=max_det,
                          retina_masks=retina, verbose=False)[0]
        stem = Path(f).stem
        if r.masks is None:
            continue
        m = r.masks.data.cpu().numpy()
        cf = r.boxes.conf.cpu().numpy()
        # Kaggle requires mutually exclusive masks per image; apply the same
        # greedy assignment here so local PQ matches what can be submitted
        cands = [((m[j] > 0.5).astype(np.uint8), float(cf[j]))
                 for j in range(m.shape[0])]
        cands.sort(key=lambda x: -x[1])
        occupied = None
        for mm, _c in cands:
            if mm.sum() == 0:
                continue
            if occupied is not None:
                mm = np.logical_and(mm, np.logical_not(occupied)).astype(np.uint8)
            if mm.sum() == 0:
                continue
            occupied = mm if occupied is None else np.logical_or(occupied, mm)
            rle = mu.encode(np.asfortranarray(mm))
            out[stem].append(rle['counts'].decode('ascii'))
        if (k + 1) % 20 == 0:
            print(f'  predicted {k+1}/{len(files)}', flush=True)
    return dict(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--imgsz', type=int, default=1280)
    ap.add_argument('--conf', type=float, default=0.001)
    ap.add_argument('--iou', type=float, default=0.7)
    ap.add_argument('--max-det', type=int, default=50)
    ap.add_argument('--retina', action='store_true')
    ap.add_argument('--split', default='val')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--save-json', default='')
    a = ap.parse_args()

    split = json.load(open(BASE / 'data/split.json'))
    files = split[a.split]
    if a.limit:
        files = files[:a.limit]
    print(f'{len(files)} {a.split} images')
    gt = build_gt(files)
    model = YOLO(a.weights)
    pred = predict_map(model, files, a.imgsz, a.conf, a.iou, a.max_det, a.retina)
    res = evaluate(gt, pred)
    print(json.dumps(res, indent=1))
    if a.save_json:
        json.dump({'result': res, 'args': vars(a)}, open(a.save_json, 'w'), indent=1)


if __name__ == '__main__':
    main()
