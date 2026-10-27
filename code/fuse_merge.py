#!/usr/bin/env python3
"""Union of two instance sources: YOLO proposals plus U-Net connected components.

Refine only moves boundaries; this adds instances the detector missed. A U-Net
component is kept when it does not already overlap a YOLO proposal, so the two
sources complement rather than duplicate each other.
"""
import argparse, json, sys
from pathlib import Path
import cv2, numpy as np, torch
from ultralytics import YOLO
import segmentation_models_pytorch as smp
from pycocotools import mask as mu

sys.path.insert(0, '/data/research/sota_method_2026-10-24/solar_filament/code')
from pq_official import evaluate
from eval_yolo_val import build_gt

BASE = Path('/data/research/sota_method_2026-10-24/solar_filament')
RAW = BASE / 'data/raw/MAGFiLO_1.0_Kaggle_2026'


def load_unet(ckpt, encoder='resnet34'):
    ck = torch.load(ckpt, map_location='cpu', weights_only=False)
    m = smp.Unet(encoder_name=encoder, encoder_weights=None, in_channels=3, classes=1)
    m.load_state_dict(ck['model']); return m.eval().cuda()


def unet_prob(model, gray, size):
    h, w = gray.shape
    small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(small.astype(np.float32) / 255.0)[None, None].repeat(1, 3, 1, 1).cuda()
    with torch.no_grad():
        p = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    return cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR)


def exclusive(items):
    items = sorted(items, key=lambda x: -x[1])
    occ = None; out = []
    for m, s in items:
        if occ is not None:
            m = np.logical_and(m, np.logical_not(occ)).astype(np.uint8)
        if m.sum() == 0:
            continue
        occ = m if occ is None else np.logical_or(occ, m)
        out.append(m)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--yolo', nargs='+', required=True)
    ap.add_argument('--unet', default=str(BASE / 'runs/unet1024/best.pt'))
    ap.add_argument('--n', type=int, default=50)
    ap.add_argument('--conf', type=float, default=0.5)
    ap.add_argument('--imgsz', type=int, default=1280)
    ap.add_argument('--uthr-list', default='0.9')
    ap.add_argument('--min-area', type=int, default=200)
    ap.add_argument('--novel-iou', type=float, default=0.3)
    a = ap.parse_args()
    split = json.load(open(BASE / 'data/split.json'))['val'][:a.n]
    gt = build_gt(split)
    ym = [YOLO(w) for w in a.yolo]; um = load_unet(a.unet)
    cache = {}
    for k, f in enumerate(split):
        gray = cv2.imread(str(RAW / 'train/train_images' / f), cv2.IMREAD_GRAYSCALE)
        props = []
        for m in ym:
            r = m.predict(str(RAW / 'train/train_images' / f), imgsz=a.imgsz, conf=a.conf,
                          iou=0.7, max_det=100, retina_masks=True, verbose=False)[0]
            if r.masks is None: continue
            ms = r.masks.data.cpu().numpy(); cf = r.boxes.conf.cpu().numpy()
            props += [((ms[j] > 0.5).astype(np.uint8), float(cf[j])) for j in range(ms.shape[0])]
        props = exclusive(props)
        cache[Path(f).stem] = (props, unet_prob(um, gray, 1024))
        if (k + 1) % 20 == 0:
            print(f'  {k+1}/{len(split)}', flush=True)
    for uthr in [float(x) for x in a.uthr_list.split(',')]:
        pm = {}
        for stem, (props, prob) in cache.items():
            u = (prob > uthr).astype(np.uint8)
            nlab, lab, stats, _ = cv2.connectedComponentsWithStats(u, 8)
            comps = [(lab == i).astype(np.uint8) for i in range(1, nlab)
                     if stats[i, cv2.CC_STAT_AREA] >= a.min_area]
            items = [(m, 1.0) for m in props]  # proposals first (kept preferentially)
            kept = list(props)
            for c in comps:
                inter = [np.logical_and(c, k).sum() / max(c.sum() + k.sum() - np.logical_and(c, k).sum(), 1)
                         for k in kept]
                if max(inter, default=0.0) < a.novel_iou:
                    items.append((c, 0.3))
                    kept.append(c)
            out = exclusive(items)
            pm[stem] = [mu.encode(np.asfortranarray(x))['counts'].decode() for x in out]
        res = evaluate(gt, pm)
        print(f'merge uthr={uthr:.2f}: PQ={res["pq"]:.4f} SQ={res["sq"]:.4f} RQ={res["rq"]:.4f} '
              f'TP={res["tp"]} FP={res["fp"]} FN={res["fn"]}', flush=True)


if __name__ == '__main__':
    main()
