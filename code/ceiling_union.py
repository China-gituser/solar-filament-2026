#!/usr/bin/env python3
"""What PQ does a perfect union detector get?

Takes the merged per-physical-image instances (union across annotator passes)
as if they were predictions, and scores them against every annotator pass with
the official metric. This is the ceiling of the 'predict every real filament'
strategy, under perfect localisation.
"""
import json, collections, sys
from pathlib import Path
import numpy as np, cv2
from pycocotools import mask as mu

sys.path.insert(0, '/data/research/sota_method_2026-10-24/solar_filament/code')
from pq_official import evaluate, rasterize

BASE = Path('/data/research/sota_method_2026-10-24/solar_filament')
RAW = BASE / 'data/raw/MAGFiLO_1.0_Kaggle_2026'


def labels_to_pred(files):
    out = {}
    for f in files:
        stem = Path(f).stem
        p = BASE / 'yolo/labels/val' / f'{stem}.txt'
        rles = []
        for line in p.read_text().splitlines():
            v = np.array(line.split()[1:], float).reshape(-1, 2) * 2048
            m = np.zeros((2048, 2048), np.uint8)
            cv2.fillPoly(m, [np.round(v).astype(np.int32)], 1)
            if m.sum() == 0:
                continue
            rles.append(mu.encode(np.asfortranarray(m))['counts'].decode())
        out[stem] = rles
    return out


def build_gt(files):
    d = json.load(open(RAW / 'train/MAGFiLO_1.0_Annotations_kaggle2026_train.json'))
    imgs = {str(i['id']): i for i in d['images']}
    want = set(files)
    gt = collections.defaultdict(list)
    for ann in d['annotations']:
        im = imgs[str(ann['image_id'])]
        if im['file_name'] not in want:
            continue
        gt[str(ann['image_id'])].append(rasterize(ann['segmentation'], im['height'], im['width']))
    for iid, im in imgs.items():
        if im['file_name'] in want:
            gt.setdefault(str(iid), [])
    return dict(gt)


if __name__ == '__main__':
    split = json.load(open(BASE / 'data/split.json'))
    import sys as _s
    for name in (_s.argv[1:] or ('val',)):
        files = split[name]
        gt = build_gt(files)
        pred = labels_to_pred(files)
        r = evaluate(gt, pred)
        n_ann = len(gt); n_img = len(pred)
        print(f'{name}: {n_img} images, {n_ann} annotator passes -> PQ={r["pq"]:.4f} SQ={r["sq"]:.4f} RQ={r["rq"]:.4f} TP={r["tp"]} FP={r["fp"]} FN={r["fn"]}')
