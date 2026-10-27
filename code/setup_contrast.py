#!/usr/bin/env python3
"""Split the contrast images into the same physical-image train/val split and
link labels, so no validation image leaks into training."""
import json, os
from pathlib import Path
BASE = Path('/data/research/sota_method_2026-10-24/solar_filament')
split = json.load(open(BASE / 'data/split.json'))
val = {Path(f).stem for f in split['val']}
train = {Path(f).stem for f in split['train']}
img = BASE / 'yolo_contrast/images'
for s in ('train', 'val'):
    (BASE / 'yolo_contrast/labels' / s).mkdir(parents=True, exist_ok=True)
# val: images already in images/val from make_contrast; link labels from yolo/labels/val
for stem in val:
    src = BASE / 'yolo/labels/val' / f'{stem}.txt'
    dst = BASE / 'yolo_contrast/labels/val' / f'{stem}.txt'
    if src.exists() and not dst.exists():
        os.symlink(src.resolve(), dst)
    # ensure image exists in images/val
    p = img / 'val' / f'{stem}.png'
    if not p.exists():
        # generated under train by make_contrast; move it
        q = img / 'train' / f'{stem}.png'
        if q.exists():
            os.replace(q, p)
# train: keep only stems in split['train']
removed = 0
for p in list((img / 'train').glob('*.png')):
    if p.stem in val:
        p.unlink(); removed += 1
for stem in train:
    src = BASE / 'yolo/labels/train' / f'{stem}.txt'
    dst = BASE / 'yolo_contrast/labels/train' / f'{stem}.txt'
    if src.exists() and not dst.exists():
        os.symlink(src.resolve(), dst)
(BASE / 'yolo_contrast/data.yaml').write_text(
    f"path: {BASE}/yolo_contrast\ntrain: images/train\nval: images/val\nnames:\n  0: filament\n")
print('train imgs', len(list((img/'train').glob('*.png'))), 'val imgs', len(list((img/'val').glob('*.png'))))
print('removed leaked', removed)
