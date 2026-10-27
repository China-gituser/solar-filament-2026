#!/usr/bin/env python3
"""PQ of the current best pipeline split by the number of annotator passes."""
import argparse, json, sys, collections
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


def load_unet(ckpt):
    ck = torch.load(ckpt, map_location='cpu', weights_only=False)
    m = smp.Unet(encoder_name='resnet34', encoder_weights=None, in_channels=3, classes=1)
    m.load_state_dict(ck['model']); return m.eval().cuda()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--yolo', default=str(BASE / 'runs/s1280/weights/best.pt'))
    ap.add_argument('--unet', default=str(BASE / 'runs/unet1024/best.pt'))
    ap.add_argument('--n', type=int, default=106)
    ap.add_argument('--conf', type=float, default=0.5)
    ap.add_argument('--uthr', type=float, default=0.85)
    a = ap.parse_args()
    split = json.load(open(BASE / 'data/split.json'))['val'][:a.n]
    d = json.load(open(RAW / 'train/MAGFiLO_1.0_Annotations_kaggle2026_train.json'))
    imgs = {str(i['id']): i for i in d['images']}
    k_of = collections.Counter(im['file_name'] for im in d['images'])
    gt_all = build_gt(split)
    ym = YOLO(a.yolo); um = load_unet(a.unet)
    ker = np.ones((13, 13), np.uint8)
    pm = {}
    for f in split:
        gray = cv2.imread(str(RAW / 'train/train_images' / f), cv2.IMREAD_GRAYSCALE)
        r = ym.predict(str(RAW / 'train/train_images' / f), imgsz=1280, conf=a.conf,
                       iou=0.7, max_det=100, retina_masks=True, verbose=False)[0]
        if r.masks is None:
            pm[Path(f).stem] = []; continue
        masks = r.masks.data.cpu().numpy(); cf = r.boxes.conf.cpu().numpy()
        small = cv2.resize(gray, (1024, 1024), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(small.astype(np.float32)/255.0)[None, None].repeat(1,3,1,1).cuda()
        with torch.no_grad():
            p = torch.sigmoid(um(x))[0,0].cpu().numpy()
        prob = cv2.resize(p, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)
        u = (prob > a.uthr).astype(np.uint8)
        items = []
        for j in range(masks.shape[0]):
            m = (masks[j] > 0.5).astype(np.uint8)
            roi = cv2.dilate(m, ker)
            cand = np.logical_and(u, roi).astype(np.uint8)
            items.append((cand if cand.sum() >= 200 else m, float(cf[j])))
        items.sort(key=lambda z: -z[1])
        occ = None; out = []
        for m, c in items:
            if occ is not None:
                m = np.logical_and(m, np.logical_not(occ)).astype(np.uint8)
            if m.sum() == 0: continue
            occ = m if occ is None else np.logical_or(occ, m)
            out.append(m)
        pm[Path(f).stem] = [mu.encode(np.asfortranarray(x))['counts'].decode() for x in out]
    for k in (1, 2, 3):
        fs = {Path(f).stem for f in split if k_of[f] == k}
        g = {iid: v for iid, v in gt_all.items() if iid.split('-',1)[1] in fs}
        p = {s: pm[s] for s in fs}
        if not g: continue
        r = evaluate(g, p)
        print(f'k={k}: {len(fs)} images, {len(g)} passes -> PQ={r["pq"]:.4f} SQ={r["sq"]:.4f} '
              f'RQ={r["rq"]:.4f} TP={r["tp"]} FP={r["fp"]} FN={r["fn"]}')


if __name__ == '__main__':
    main()
