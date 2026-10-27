#!/usr/bin/env python3
"""Build 3-channel inputs whose channels carry the signal filaments actually have.

R = raw grayscale (context), G = large-scale local contrast, B = small-scale
local contrast. The hand-crafted baseline that scores on this competition uses
exactly this quantity, so a detector that sees it directly should need less
capacity to find faint filaments.
"""
import argparse, glob, os
from pathlib import Path
import cv2, numpy as np

BASE = Path('/data/research/sota_method_2026-10-24/solar_filament')
RAW = BASE / 'data/raw/MAGFiLO_1.0_Kaggle_2026'


def contrast(img, sigma):
    x = img.astype(np.float32) / 255.0
    smooth = cv2.GaussianBlur(x, (0, 0), 1.0)
    bg = cv2.GaussianBlur(smooth, (0, 0), sigma)
    c = np.clip((bg - smooth) / np.maximum(bg, 0.08), 0, 1)
    return (c * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--size', type=int, default=2048)
    a = ap.parse_args()
    for split, src in (('train', RAW / 'train/train_images'), ('val', RAW / 'train/train_images'),
                       ('test', RAW / 'test/test_images')):
        outdir = BASE / 'yolo_contrast' / 'images' / split
        outdir.mkdir(parents=True, exist_ok=True)
        files = sorted(glob.glob(str(src / '*.jpeg')))
        if split == 'test':
            pass
        n = 0
        for f in files:
            stem = Path(f).stem
            dst = outdir / f'{stem}.png'
            if dst.exists():
                continue
            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            g = contrast(img, 64.0)
            b = contrast(img, 12.0)
            cv2.imwrite(str(dst), cv2.merge([img, g, b]))
            n += 1
            if n % 100 == 0:
                print(f'{split}: {n} written', flush=True)
        print(f'{split}: done, {len(files)} files', flush=True)


if __name__ == '__main__':
    main()
