#!/usr/bin/env python3
"""Fuse YOLO instance proposals with the U-Net semantic probability map.

YOLO gives high-recall instance proposals with coarse boundaries; the U-Net
gives a full-resolution probability field. Refine mode keeps each proposal's
region of interest but re-thresholds it against the U-Net field, which can only
move the boundary -- not invent or delete whole instances.
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


def contrast3(gray):
    x = gray.astype(np.float32) / 255.0
    smooth = cv2.GaussianBlur(x, (0, 0), 1.0)
    out = [gray]
    for sig in (64.0, 12.0):
        bg = cv2.GaussianBlur(smooth, (0, 0), sig)
        c = np.clip((bg - smooth) / np.maximum(bg, 0.08), 0, 1)
        out.append((c * 255).astype(np.uint8))
    return cv2.merge(out)


def unet_prob(model, gray, size):
    h, w = gray.shape
    small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(small.astype(np.float32) / 255.0)[None, None].repeat(1, 3, 1, 1).cuda()
    with torch.no_grad():
        p = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    return cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR)


def exclusive_keep(masks, scores):
    order = np.argsort(-np.asarray(scores))
    occ = None; om = []; os_ = []
    for i in order:
        m = masks[i].copy()
        if occ is not None:
            m = np.logical_and(m, np.logical_not(occ)).astype(np.uint8)
        if m.sum() == 0:
            continue
        occ = m if occ is None else np.logical_or(occ, m)
        om.append(m); os_.append(scores[i])
    return om, os_


def exclusive(masks, scores):
    order = np.argsort(-np.asarray(scores))
    occ = None; out = []
    for i in order:
        m = masks[i].copy()
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
    ap.add_argument('--conf', type=float, default=0.4)
    ap.add_argument('--imgsz', type=int, default=1280)
    ap.add_argument('--unet-size', type=int, default=1024)
    ap.add_argument('--modes', default='none,intersect,union,refine')
    ap.add_argument('--uthr-list', default='')
    ap.add_argument('--dilate-list', default='')
    ap.add_argument('--flip', action='store_true')
    ap.add_argument('--conf-list', default='')
    ap.add_argument('--conf-low', type=float, default=0.0)
    ap.add_argument('--contrast', action='store_true',
                    help='feed the detector [gray, coarse contrast, fine contrast]')
    ap.add_argument('--uthr', type=float, default=0.7)
    ap.add_argument('--dilate', type=int, default=6)
    ap.add_argument('--min-area', type=int, default=200)
    a = ap.parse_args()
    split = json.load(open(BASE / 'data/split.json'))['val'][:a.n]
    gt = build_gt(split)
    ym = [YOLO(w) for w in a.yolo]; um = load_unet(a.unet)
    cache = {}
    for k, f in enumerate(split):
        gray = cv2.imread(str(RAW / 'train/train_images' / f), cv2.IMREAD_GRAYSCALE)
        pair = []
        thr = a.conf_low if a.conf_list else a.conf
        det_in = contrast3(gray) if a.contrast else str(RAW / 'train/train_images' / f)
        for net in ym:
            r = net.predict(det_in, imgsz=a.imgsz, conf=thr,
                            iou=0.7, max_det=100, retina_masks=True, verbose=False)[0]
            if r.masks is None:
                continue
            ms = r.masks.data.cpu().numpy(); cf = r.boxes.conf.cpu().numpy()
            pair += [((ms[j] > 0.5).astype(np.uint8), float(cf[j])) for j in range(ms.shape[0])]
        if a.flip:
            rf = ym.predict(np.ascontiguousarray(gray[:, ::-1]), imgsz=a.imgsz, conf=a.conf,
                            iou=0.7, max_det=100, retina_masks=True, verbose=False)[0]
            if rf.masks is not None:
                for j in range(rf.masks.data.shape[0]):
                    pair.append(((rf.masks.data.cpu().numpy()[j] > 0.5).astype(np.uint8)[:, ::-1],
                                 float(rf.boxes.conf.cpu().numpy()[j])))
        if pair:
            pair.sort(key=lambda x: -x[1])
            cached_m, cached_s = exclusive_keep([m for m, _ in pair], [c for _, c in pair])
        else:
            cached_m, cached_s = [], []
        cache[Path(f).stem] = (cached_m, cached_s, unet_prob(um, gray, a.unet_size))
        if (k + 1) % 20 == 0:
            print(f'  {k+1}/{len(split)}', flush=True)
    import itertools
    clist = [float(x) for x in a.conf_list.split(',')] if a.conf_list else [a.conf]
    grid = []
    for u in ([float(x) for x in a.uthr_list.split(',')] if a.uthr_list else [a.uthr]):
        for d in ([int(x) for x in a.dilate_list.split(',')] if a.dilate_list else [a.dilate]):
            for c in clist:
                grid.append((u, d, c))
    for uthr, dil, cthr in grid:
        ker = np.ones((dil * 2 + 1,) * 2, np.uint8)
        pm = {}
        for stem, (masks, scores, prob) in cache.items():
            u = (prob > uthr).astype(np.uint8)
            sel = [m for m, sc in zip(masks, scores) if sc >= cthr]
            out = []
            for m in sel:
                roi = cv2.dilate(m, ker)
                cand = np.logical_and(u, roi).astype(np.uint8)
                x = cand if cand.sum() >= a.min_area else m
                if x.sum() == 0:
                    continue
                out.append(x)
            out = exclusive(out, list(range(len(out)))[::-1])
            pm[stem] = [mu.encode(np.asfortranarray(x))['counts'].decode() for x in out]
        res = evaluate(gt, pm)
        print(f'refine conf={cthr:.2f} uthr={uthr:.2f} d={dil}: PQ={res["pq"]:.4f} '
              f'SQ={res["sq"]:.4f} RQ={res["rq"]:.4f} TP={res["tp"]} FP={res["fp"]} '
              f'FN={res["fn"]} n={sum(len(v) for v in pm.values())}', flush=True)


if __name__ == '__main__':
    main()
