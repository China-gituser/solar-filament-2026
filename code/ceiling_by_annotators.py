#!/usr/bin/env python3
"""Union-oracle PQ split by how many annotator passes a physical image has."""
import json, collections, sys
from pathlib import Path
import numpy as np, cv2
from pycocotools import mask as mu

sys.path.insert(0, '/data/research/sota_method_2026-10-24/solar_filament/code')
from pq_official import evaluate

BASE = Path('/data/research/sota_method_2026-10-24/solar_filament')
RAW = BASE / 'data/raw/MAGFiLO_1.0_Kaggle_2026'


def main(split='val'):
    files = json.load(open(BASE / 'data/split.json'))[split]
    want = set(files)
    d = json.load(open(RAW / 'train/MAGFiLO_1.0_Annotations_kaggle2026_train.json'))
    imgs = {str(i['id']): i for i in d['images']}
    by_file = collections.defaultdict(dict)
    for ann in d['annotations']:
        im = imgs[str(ann['image_id'])]
        if im['file_name'] not in want:
            continue
        by_file[im['file_name']].setdefault(str(ann['image_id']), []).append(ann)
    cache = json.load(open(BASE / 'data/gt_rle_train.json'))
    gt = {iid: cache.get(iid, []) for iid, im in imgs.items() if im['file_name'] in want}
    # union oracle from cached GT rles: merge masks across annotators by IoU>0.5
    from pq_official import decode
    def union_for(f):
        masks = []
        for iid in by_file[f]:
            for r in cache.get(iid, []):
                masks.append(decode([r])[0].astype(np.uint8))
        if not masks:
            return []
        n = len(masks); parent = list(range(n))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x
        for i in range(n):
            for j in range(i + 1, n):
                inter = np.logical_and(masks[i], masks[j]).sum()
                if inter == 0: continue
                a, b = masks[i].sum(), masks[j].sum()
                if inter / (a + b - inter) > 0.5:
                    ri, rj = find(i), find(j)
                    if ri != rj: parent[rj] = ri
        groups = collections.defaultdict(list)
        for i in range(n): groups[find(i)].append(i)
        out = []
        for g in groups.values():
            m = masks[g[0]].copy()
            for i in g[1:]: m |= masks[i]
            out.append(mu.encode(np.asfortranarray(m))['counts'].decode())
        return out
    groups = collections.defaultdict(list)
    for f in files:
        k = len(by_file.get(f, {}))
        groups[k].append(f)
    for k in sorted(groups):
        fs = groups[k]
        g = {iid: cache.get(iid, []) for iid, im in imgs.items() if im['file_name'] in set(fs)}
        p = {Path(f).stem: union_for(f) for f in fs}
        r = evaluate(g, p)
        print(f'{k} annotator(s): {len(fs):3d} images, {len(g):3d} passes -> '
              f'oracle PQ={r["pq"]:.4f} SQ={r["sq"]:.4f} RQ={r["rq"]:.4f}')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'val')
