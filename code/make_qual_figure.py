#!/usr/bin/env python3
"""Qualitative panel: H-alpha image, ground truth, and refined prediction."""
import json, sys
from pathlib import Path
import cv2, numpy as np, torch
from ultralytics import YOLO
import segmentation_models_pytorch as smp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path('/data/research/sota_method_2026-10-24/solar_filament')
RAW = BASE / 'data/raw/MAGFiLO_1.0_Kaggle_2026'

def main():
    split = json.load(open(BASE / 'data/split.json'))['val']
    d = json.load(open(RAW / 'train/MAGFiLO_1.0_Annotations_kaggle2026_train.json'))
    imgs = {str(i['id']): i for i in d['images']}
    # pick an image with >= 5 annotations
    cnt = {}
    for a in d['annotations']:
        cnt[str(a['image_id'])] = cnt.get(str(a['image_id']), 0) + 1
    target = None
    for f in split:
        for iid, im in imgs.items():
            if im['file_name'] == f and cnt.get(iid, 0) >= 5:
                target = (f, iid); break
        if target: break
    f, iid = target
    gray = cv2.imread(str(RAW / 'train/train_images' / f), cv2.IMREAD_GRAYSCALE)
    gt = np.zeros_like(gray)
    for a in d['annotations']:
        if str(a['image_id']) == iid:
            for poly in a['segmentation']:
                cv2.fillPoly(gt, [np.round(np.asarray(poly).reshape(-1, 2)).astype(np.int32)], 255)
    ym = YOLO(str(BASE / 'runs/s1280/weights/best.pt'))
    ck = torch.load(BASE / 'runs/unet1024/best.pt', map_location='cpu', weights_only=False)
    um = smp.Unet(encoder_name='resnet34', encoder_weights=None, in_channels=3, classes=1)
    um.load_state_dict(ck['model']); um.eval().cuda()
    r = ym.predict(str(RAW / 'train/train_images' / f), imgsz=1280, conf=0.5,
                   iou=0.7, retina_masks=True, verbose=False)[0]
    small = cv2.resize(gray, (1024, 1024), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(small.astype(np.float32)/255.0)[None,None].repeat(1,3,1,1).cuda()
    with torch.no_grad():
        p = torch.sigmoid(um(x))[0,0].cpu().numpy()
    prob = cv2.resize(p, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)
    u = (prob > 0.85).astype(np.uint8)
    ker = np.ones((13,13), np.uint8)
    pred = np.zeros_like(gray)
    if r.masks is not None:
        ms = r.masks.data.cpu().numpy()
        for j in range(ms.shape[0]):
            m = (ms[j] > 0.5).astype(np.uint8)
            roi = cv2.dilate(m, ker)
            cand = np.logical_and(u, roi).astype(np.uint8)
            pred[np.logical_or(pred > 0, (cand if cand.sum() >= 200 else m) > 0)] = 255
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.5))
    ax[0].imshow(gray, cmap='gray'); ax[0].set_title('H-alpha', fontsize=8)
    ax[1].imshow(gray, cmap='gray'); ax[1].imshow(np.ma.masked_where(gt == 0, gt), cmap='spring', alpha=0.45)
    ax[1].set_title('ground truth', fontsize=8)
    ax[2].imshow(gray, cmap='gray'); ax[2].imshow(np.ma.masked_where(pred == 0, pred), cmap='winter', alpha=0.45)
    ax[2].set_title('refined prediction', fontsize=8)
    for a in ax: a.axis('off')
    plt.tight_layout(pad=0.3)
    plt.savefig(BASE / 'paper/qualitative.pdf', bbox_inches='tight')
    print('saved', f)

main()
