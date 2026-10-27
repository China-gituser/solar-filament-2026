#!/usr/bin/env python3
"""Rasterise the train GT once and cache per annotator-image RLE strings."""
import json, collections, sys
from pathlib import Path
from pycocotools import mask as mu
import numpy as np

BASE = Path('/data/research/sota_method_2026-10-24/solar_filament')
RAW = BASE / 'data/raw/MAGFiLO_1.0_Kaggle_2026'


def main():
    d = json.load(open(RAW / 'train/MAGFiLO_1.0_Annotations_kaggle2026_train.json'))
    imgs = {str(i['id']): i for i in d['images']}
    out = collections.defaultdict(list)
    for ann in d['annotations']:
        im = imgs[str(ann['image_id'])]
        rle = mu.merge(mu.frPyObjects(ann['segmentation'], im['height'], im['width']))
        out[str(ann['image_id'])].append(rle['counts'].decode('ascii'))
    for iid in imgs:
        out.setdefault(str(iid), [])
    path = BASE / 'data/gt_rle_train.json'
    json.dump(dict(out), open(path, 'w'))
    print('cached', len(out), 'annotator-image entries ->', path)


if __name__ == '__main__':
    main()
