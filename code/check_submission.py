#!/usr/bin/env python3
"""Validate a submission CSV against the competition's stated contract."""
import argparse, csv, collections
import numpy as np
from pycocotools import mask as mu

ap = argparse.ArgumentParser()
ap.add_argument('path')
ap.add_argument('--test-dir', default='/data/research/sota_method_2026-10-24/solar_filament/data/raw/MAGFiLO_1.0_Kaggle_2026/test/test_images')
a = ap.parse_args()

import os
stems = {os.path.splitext(f)[0] for f in os.listdir(a.test_dir)}
rows = list(csv.DictReader(open(a.path)))
print('rows', len(rows), 'header ok', list(rows[0].keys()) if rows else None)
bad = 0; by_img = collections.defaultdict(int); dup = collections.Counter()
for r in rows:
    fid = r['filament_id']; dup[fid] += 1
    if '_' not in fid:
        bad += 1; continue
    stem = fid.split('_', 1)[0]
    if stem not in stems:
        bad += 1
    by_img[stem] += 1
    try:
        m = mu.decode({'size': [2048, 2048], 'counts': r['segmentation_rle'].encode()})
        if m.shape != (2048, 2048) or m.sum() == 0:
            bad += 1
    except Exception:
        bad += 1
print('images covered', len(by_img), '/', len(stems))
print('bad ids or RLE', bad)
print('duplicate filament_id', sum(1 for v in dup.values() if v > 1))
print('preds per image: mean %.2f max %d' % (len(rows)/max(len(by_img),1), max(by_img.values()) if by_img else 0))
# overlap check: masks of the same image must be mutually exclusive
import numpy as np
acc = collections.defaultdict(lambda: np.zeros(2048*2048, np.uint8))
overlaps = 0
for r in rows:
    stem = r['filament_id'].split('_', 1)[0]
    m = mu.decode({'size': [2048, 2048], 'counts': r['segmentation_rle'].encode()}).reshape(-1).astype(np.uint8)
    if acc[stem].sum() and np.logical_and(acc[stem], m).any():
        overlaps += 1
    acc[stem] = np.logical_or(acc[stem], m)
print('masks overlapping another mask in the same image:', overlaps)
