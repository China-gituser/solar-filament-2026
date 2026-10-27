#!/usr/bin/env python3
"""Turn U-Net semantic predictions into instances (connected components) and
score them with the official PQ.

Connected components are mutually exclusive by construction, which satisfies
Kaggle's no-overlap rule without any greedy assignment.
"""
import argparse, json, sys
from pathlib import Path

import cv2, numpy as np, torch
import segmentation_models_pytorch as smp
from pycocotools import mask as mu

sys.path.insert(0, '/data/research/sota_method_2026-10-24/solar_filament/code')
from pq_official import evaluate
from eval_yolo_val import build_gt

BASE = Path('/data/research/sota_method_2026-10-24/solar_filament')
RAW = BASE / 'data/raw/MAGFiLO_1.0_Kaggle_2026'


def load_model(ckpt_path, encoder):
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model = smp.Unet(encoder_name=encoder, encoder_weights=None, in_channels=3, classes=1)
    model.load_state_dict(ck['model'])
    model.eval().cuda()
    return model


def predict(model, img_path, size):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    small = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(small.astype(np.float32) / 255.0)[None, None].repeat(1, 3, 1, 1).cuda()
    with torch.no_grad():
        prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    return cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)


def to_instances(prob, thr, min_area):
    binary = (prob > thr).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    out = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        out.append((lab == i).astype(np.uint8))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--encoder', default='resnet34')
    ap.add_argument('--size', type=int, default=1024)
    ap.add_argument('--n', type=int, default=50)
    ap.add_argument('--thrs', default='0.2,0.3,0.4,0.5,0.6,0.7')
    ap.add_argument('--min-area', type=int, default=200)
    ap.add_argument('--save-probs', default='')
    a = ap.parse_args()
    split = json.load(open(BASE / 'data/split.json'))['val'][:a.n]
    gt = build_gt(split)
    model = load_model(a.ckpt, a.encoder)
    probs = {}
    for k, f in enumerate(split):
        probs[Path(f).stem] = predict(model, RAW / 'train/train_images' / f, a.size)
        if (k + 1) % 20 == 0:
            print(f'  {k+1}/{len(split)}', flush=True)
    if a.save_probs:
        np.savez_compressed(a.save_probs, **probs)
    for thr in [float(x) for x in a.thrs.split(',')]:
        pm = {}
        for stem, p in probs.items():
            rles = []
            for m in to_instances(p, thr, a.min_area):
                rles.append(mu.encode(np.asfortranarray(m))['counts'].decode())
            pm[stem] = rles
        res = evaluate(gt, pm)
        print(f'thr={thr:.2f}: PQ={res["pq"]:.4f} SQ={res["sq"]:.4f} RQ={res["rq"]:.4f} '
              f'TP={res["tp"]} FP={res["fp"]} FN={res["fn"]} n_pred={sum(len(v) for v in pm.values())}', flush=True)


if __name__ == '__main__':
    main()
