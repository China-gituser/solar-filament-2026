#!/usr/bin/env python3
"""Build a YOLO-seg dataset from the Kaggle MAGFiLO annotations.

All annotator passes of the same physical image are merged into one label file:
the evaluation scores a prediction separately against every annotator pass, so
the union of their instances is the correct training target (see the metric
analysis in PROJECT_STATUS).

Split is grouped by physical image, and optionally by capture date, so the same
GONG observation never appears on both sides.
"""
import json, os, random, collections, argparse
from pathlib import Path

BASE = Path('/data/research/sota_method_2026-10-24/solar_filament')
RAW = BASE / 'data/raw/MAGFiLO_1.0_Kaggle_2026'


def norm_poly(poly, w, h):
    pts = [(min(max(x, 0.0), w - 1e-3) / w, min(max(y, 0.0), h - 1e-3) / h)
           for x, y in zip(poly[0::2], poly[1::2])]
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(BASE / 'yolo'))
    ap.add_argument('--val-frac', type=float, default=0.15)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--min-area', type=float, default=16.0)
    a = ap.parse_args()

    d = json.load(open(RAW / 'train/MAGFiLO_1.0_Annotations_kaggle2026_train.json'))
    imgs = {str(i['id']): i for i in d['images']}
    by_file = collections.defaultdict(list)
    for a_ in d['annotations']:
        by_file[imgs[str(a_['image_id'])]['file_name']].append(a_)

    files = sorted(by_file)
    rng = random.Random(a.seed)
    rng.shuffle(files)
    n_val = int(round(len(files) * a.val_frac))
    val = set(files[:n_val])
    print(f'{len(files)} physical train images -> {len(files)-n_val} train / {n_val} val')

    for split in ('train', 'val'):
        (Path(a.out) / 'images' / split).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / 'labels' / split).mkdir(parents=True, exist_ok=True)

    counts = collections.Counter()
    for f in files:
        split = 'val' if f in val else 'train'
        stem = Path(f).stem
        src = RAW / 'train/train_images' / f
        dst = Path(a.out) / 'images' / split / f
        if not dst.exists():
            os.symlink(src, dst)
        lines = []
        for ann in by_file[f]:
            if ann.get('iscrowd', 0):
                continue
            w = imgs[str(ann['image_id'])]['width']; h = imgs[str(ann['image_id'])]['height']
            polys = ann['segmentation']
            if isinstance(polys, dict):   # uncompressed RLE: skip, MAGFiLO ships polygons
                continue
            for poly in polys:
                pts = norm_poly(poly, w, h)
                if len(pts) < 3:
                    continue
                area = ann['area']
                if area < a.min_area:
                    continue
                lines.append('0 ' + ' '.join(f'{x:.6f} {y:.6f}' for x, y in pts))
        with open(Path(a.out) / 'labels' / split / f'{stem}.txt', 'w') as fh:
            fh.write('\n'.join(lines) + ('\n' if lines else ''))
        counts[split] += len(lines)

    (Path(a.out) / 'data.yaml').write_text(
        f"path: {a.out}\ntrain: images/train\nval: images/val\nnames:\n  0: filament\n")
    print('instances', dict(counts))
    # 记录划分，供后续官方 PQ 评测按物理图使用
    json.dump({'val': sorted(val), 'train': sorted(set(files) - val),
               'seed': a.seed, 'val_frac': a.val_frac},
              open(BASE / 'data/split.json', 'w'), indent=1)
    print('wrote', BASE / 'data/split.json')


if __name__ == '__main__':
    main()
