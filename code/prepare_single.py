#!/usr/bin/env python3
"""Dataset where every annotator pass is its own training sample.

The merged-union labels make the detector over-predict (mean area 2513 vs GT
1821). Training on the individual expert annotations instead teaches the actual
annotation shape. The same JPEG is symlinked once per annotator pass under
"<annotator_id>-<image_id>.jpeg".
"""
import json, os, random, collections
from pathlib import Path

BASE = Path('/data/research/sota_method_2026-10-24/solar_filament')
RAW = BASE / 'data/raw/MAGFiLO_1.0_Kaggle_2026'
OUT = BASE / 'yolo_single'
VAL_FRAC = 0.15
SEED = 42


def main():
    d = json.load(open(RAW / 'train/MAGFiLO_1.0_Annotations_kaggle2026_train.json'))
    imgs = {str(i['id']): i for i in d['images']}
    by_ann_img = collections.defaultdict(list)
    for ann in d['annotations']:
        by_ann_img[str(ann['image_id'])].append(ann)
    physical = sorted({im['file_name'] for im in d['images']})
    rng = random.Random(SEED); rng.shuffle(physical)
    n_val = int(round(len(physical) * VAL_FRAC))
    val = set(physical[:n_val])
    for split in ('train', 'val'):
        (OUT / 'images' / split).mkdir(parents=True, exist_ok=True)
        (OUT / 'labels' / split).mkdir(parents=True, exist_ok=True)
    counts = collections.Counter()
    n_samples = collections.Counter()
    for iid, im in imgs.items():
        split = 'val' if im['file_name'] in val else 'train'
        name = f'{iid}.jpeg'
        dst = OUT / 'images' / split / name
        if not dst.exists():
            os.symlink(RAW / 'train/train_images' / im['file_name'], dst)
        lines = []
        for ann in by_ann_img.get(iid, []):
            if ann.get('iscrowd', 0):
                continue
            w, h = im['width'], im['height']
            for poly in ann['segmentation']:
                if len(poly) < 6:
                    continue
                xs = poly[0::2]; ys = poly[1::2]
                pts = [(min(max(x, 0) / w, 0.999999), min(max(y, 0) / h, 0.999999))
                       for x, y in zip(xs, ys)]
                lines.append('0 ' + ' '.join(f'{x:.6f} {y:.6f}' for x, y in pts))
        (OUT / 'labels' / split / f'{iid}.txt').write_text(
            '\n'.join(lines) + ('\n' if lines else ''))
        counts[split] += len(lines); n_samples[split] += 1
    (OUT / 'data.yaml').write_text(
        f"path: {OUT}\ntrain: images/train\nval: images/val\nnames:\n  0: filament\n")
    print('samples', dict(n_samples), 'instances', dict(counts))
    print('val physical images', len(val))


if __name__ == '__main__':
    main()
