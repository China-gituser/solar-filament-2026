#!/usr/bin/env python3
"""Test-set prediction: YOLO proposals refined by the U-Net probability field,
then made mutually exclusive, then written as the submission CSV."""
import argparse, csv
from pathlib import Path
import cv2, numpy as np, torch
from ultralytics import YOLO
import segmentation_models_pytorch as smp
from pycocotools import mask as mu

BASE = Path('/data/research/sota_method_2026-10-24/solar_filament')


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--yolo', nargs='+', required=True)
    ap.add_argument('--unet', default=str(BASE / 'runs/unet1024/best.pt'))
    ap.add_argument('--imgsz', type=int, default=1280)
    ap.add_argument('--unet-size', type=int, default=1024)
    ap.add_argument('--conf', type=float, default=0.5)
    ap.add_argument('--uthr', type=float, default=0.85)
    ap.add_argument('--dilate', type=int, default=6)
    ap.add_argument('--min-area', type=int, default=200)
    ap.add_argument('--contrast', action='store_true')
    ap.add_argument('--test-dir', default=str(BASE / 'data/raw/MAGFiLO_1.0_Kaggle_2026/test/test_images'))
    ap.add_argument('--out', default=str(BASE / 'out/sub_refine.csv'))
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    files = sorted(Path(a.test_dir).glob('*.jpeg')) + sorted(Path(a.test_dir).glob('*.jpg'))
    if a.limit:
        files = files[:a.limit]
    ym = [YOLO(w) for w in a.yolo]
    um = load_unet(a.unet)
    ker = np.ones((a.dilate * 2 + 1,) * 2, np.uint8)
    rows = []; empty = 0
    for k, f in enumerate(files):
        gray = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        cands = []
        det_in = contrast3(gray) if a.contrast else str(f)
        for m in ym:
            r = m.predict(det_in, imgsz=a.imgsz, conf=a.conf, iou=0.7, max_det=100,
                          retina_masks=True, verbose=False)[0]
            if r.masks is None:
                continue
            masks = r.masks.data.cpu().numpy(); cf = r.boxes.conf.cpu().numpy()
            for j in range(masks.shape[0]):
                cands.append(((masks[j] > 0.5).astype(np.uint8), float(cf[j])))
        if not cands:
            empty += 1
            continue
        prob = unet_prob(um, gray, a.unet_size)
        u = (prob > a.uthr).astype(np.uint8)
        refined = []
        for mm, c in cands:
            roi = cv2.dilate(mm, ker)
            cand = np.logical_and(u, roi).astype(np.uint8)
            refined.append((cand if cand.sum() >= a.min_area else mm, c))
        refined.sort(key=lambda x: -x[1])
        occ = None; kept = []
        for mm, c in refined:
            if occ is not None:
                mm = np.logical_and(mm, np.logical_not(occ)).astype(np.uint8)
            if mm.sum() == 0:
                continue
            occ = mm if occ is None else np.logical_or(occ, mm)
            kept.append(mm)
        if not kept:
            empty += 1
        for i, mm in enumerate(kept):
            rows.append({'filament_id': f'{f.stem}_{i}',
                         'segmentation_rle': mu.encode(np.asfortranarray(mm))['counts'].decode()})
        if (k + 1) % 20 == 0:
            print(f'{k+1}/{len(files)} rows={len(rows)} empty={empty}', flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['filament_id', 'segmentation_rle'])
        w.writeheader(); w.writerows(rows)
    print(f'wrote {a.out}: {len(rows)} rows over {len(files)} images, {empty} empty')


if __name__ == '__main__':
    main()
