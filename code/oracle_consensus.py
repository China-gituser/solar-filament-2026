#!/usr/bin/env python3
"""Oracle analysis: which subset of real instances maximises the official PQ?

For every physical validation image we take the union instances and, for each,
count how many annotator passes contain a matching instance (IoU > 0.5). Then we
score official PQ for prediction sets filtered at different consensus levels.
This upper-bounds any detector and tells us what the metric actually rewards.
"""
import json, collections, sys
from pathlib import Path
import numpy as np, cv2
from pycocotools import mask as mu

sys.path.insert(0, '/data/research/sota_method_2026-10-24/solar_filament/code')
from pq_official import evaluate, rasterize

BASE = Path('/data/research/sota_method_2026-10-24/solar_film filament'.replace('film filament','filament'))
RAW = BASE / 'data/raw/MAGFiLO_1.0_Kaggle_2026'


def rle(mask):
    return mu.encode(np.asfortranarray(mask))['counts'].decode()


def poly_to_mask(poly):
    m = np.zeros((2048, 2048), np.uint8)
    v = np.asarray(poly, float).reshape(-1, 2)
    cv2.fillPoly(m, [np.round(v).astype(np.int32)], 1)
    return m


def main(split='val'):
    split_files = json.load(open(BASE / 'data/split.json'))[split]
    want = set(split_files)
    d = json.load(open(RAW / 'train/MAGFiLO_1.0_Annotations_kaggle2026_train.json'))
    imgs = {str(i['id']): i for i in d['images']}
    per_file = collections.defaultdict(lambda: collections.defaultdict(list))  # file -> ann_img -> [masks]
    for ann in d['annotations']:
        im = imgs[str(ann['image_id'])]
        if im['file_name'] not in want:
            continue
        for poly in ann['segmentation']:
            if len(poly) >= 6:
                per_file[im['file_name']][str(ann['image_id'])].append(poly_to_mask(poly))

    gt = {}
    for f, by_ann in per_file.items():
        for ann_img, masks in by_ann.items():
            gt[ann_img] = [rle(m) for m in masks]
    # images with no annotation still count as empty entries
    for iid, im in imgs.items():
        if im['file_name'] in want:
            gt.setdefault(str(iid), [])

    # union instances per physical image (cluster across annotators)
    union_pred = {}
    for f, by_ann in per_file.items():
        stem = Path(f).stem
        items = [(ann_img, m) for ann_img, ms in by_ann.items() for m in ms]
        if not items:
            union_pred[stem] = []
            continue
        n = len(items)
        parent = list(range(n))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x
        for i in range(n):
            for j in range(i + 1, n):
                if items[i][0] == items[j][0]:
                    continue  # same annotator: never merge two of their own instances
                inter = np.logical_and(items[i][1], items[j][1]).sum()
                if inter == 0: continue
                a, b = items[i][1].sum(), items[j][1].sum()
                if inter / (a + b - inter) > 0.5:
                    ri, rj = find(i), find(j)
                    if ri != rj: parent[rj] = ri
        groups = collections.defaultdict(list)
        for i in range(n): groups[find(i)].append(i)
        union_pred[stem] = []
        for g in groups.values():
            m = items[g[0]][1].copy()
            for i in g[1:]: m |= items[i][1]
            union_pred[stem].append((m, len(set(items[i][0] for i in g))))

    for level in range(1, 4):
        pred = {stem: [rle(m) for m, k in items if k >= level]
                for stem, items in union_pred.items()}
        r = evaluate(gt, pred)
        print(f'keep instances covered by >= {level} annotator(s): '
              f'PQ={r["pq"]:.4f} SQ={r["sq"]:.4f} RQ={r["rq"]:.4f} '
              f'TP={r["tp"]} FP={r["fp"]} FN={r["fn"]}')
    # distribution of coverage
    dist = collections.Counter(k for items in union_pred.values() for _, k in items)
    print('union instance coverage histogram', dict(sorted(dist.items())))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'val')
