#!/usr/bin/env python3
"""Merge duplicate instances across annotator passes of the same GONG image.

Two annotators tracing the same filament produce two polygons. Trained as-is,
they teach the detector that one object is two objects. We cluster instances by
mask IoU and keep the union of each cluster as a single target, because the
metric scores the prediction separately against every annotator pass: covering
the union is worth more than matching one pass exactly.

Writes YOLO polygon labels plus an audit of how many duplicates were merged.
"""
import json, collections, argparse
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mu

BASE = Path('/data/research/sota_method_2026-10-24/solar_filament')
RAW = BASE / 'data/raw/MAGFiLO_1.0_Kaggle_2026'


def poly_mask(poly, h=2048, w=2048):
    m = np.zeros((h, w), np.uint8)
    p = np.asarray(poly, np.float64).reshape(-1, 2)
    cv2.fillPoly(m, [np.round(p).astype(np.int32)], 1)
    return m


def cluster_merge(masks, iou_thr=0.6):
    """Greedy union-find over pairwise IoU; returns merged binary masks."""
    n = len(masks)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    areas = [int(m.sum()) for m in masks]
    for i in range(n):
        for j in range(i + 1, n):
            inter = int(np.logical_and(masks[i], masks[j]).sum())
            if inter == 0:
                continue
            iou = inter / (areas[i] + areas[j] - inter)
            if iou > iou_thr:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri
    groups = collections.defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    out = []
    for g in groups.values():
        m = masks[g[0]].copy()
        for i in g[1:]:
            m |= masks[i]
        out.append((m, g))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--iou', type=float, default=0.6)
    ap.add_argument('--min-area', type=float, default=40.0)
    ap.add_argument('--split-dir', default=str(BASE / 'yolo'))
    a = ap.parse_args()

    d = json.load(open(RAW / 'train/MAGFiLO_1.0_Annotations_kaggle2026_train.json'))
    imgs = {str(i['id']): i for i in d['images']}
    by_file = collections.defaultdict(list)
    for ann in d['annotations']:
        by_file[imgs[str(ann['image_id'])]['file_name']].append(ann)

    stats = collections.Counter()
    for f, anns in by_file.items():
        stem = Path(f).stem
        split = 'val' if (BASE / 'yolo/labels/val' / f'{stem}.txt').exists() else 'train'
        if not (BASE / 'yolo/labels' / split / f'{stem}.txt').exists():
            continue
        masks = []
        for ann in anns:
            if ann.get('iscrowd', 0):
                continue
            for poly in ann['segmentation']:
                if len(poly) < 6:
                    continue
                m = poly_mask(poly)
                if m.sum() >= a.min_area:
                    masks.append(m)
        if not masks:
            open(BASE / 'yolo/labels' / split / f'{stem}.txt', 'w').write('')
            continue
        merged = cluster_merge(masks, a.iou)
        lines = []
        for m, g in merged:
            if m.sum() < a.min_area:
                continue
            cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cs:
                continue
            c = max(cs, key=cv2.contourArea)
            ap2 = cv2.approxPolyDP(c, 1.5, True).reshape(-1, 2).astype(float)
            if len(ap2) < 3:
                continue
            ap2[:, 0] /= 2048.0; ap2[:, 1] /= 2048.0
            lines.append('0 ' + ' '.join(f'{x:.6f} {y:.6f}' for x, y in ap2))
        open(BASE / 'yolo/labels' / split / f'{stem}.txt', 'w').write(
            '\n'.join(lines) + ('\n' if lines else ''))
        stats[split] += len(lines)
        stats[split + '_raw'] += len(masks)
        stats[split + '_dup'] += len(masks) - len(merged)
    print('merged instances', dict(stats))
    print('duplicate rate train: %.1f%%' % (100 * stats['train_dup'] / max(stats['train_raw'], 1)))


if __name__ == '__main__':
    main()
