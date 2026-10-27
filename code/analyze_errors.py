#!/usr/bin/env python3
"""Where do misses and false positives come from?

Matches the current pipeline's predictions to ground truth (IoU>0.5) and
summarises FN/TP by instance size, radial position on the solar disk, and mean
local contrast. Systematic structure points to a specific fix.
"""
import json, sys, collections
from pathlib import Path
import cv2, numpy as np, torch
from ultralytics import YOLO
import segmentation_models_pytorch as smp
from pycocotools import mask as mu

sys.path.insert(0, '/data/research/sota_method_2026-10-24/solar_filament/code')
from pq_official import evaluate, decode

BASE = Path('/data/research/sota_method_2026-10-24/solar_filament')
RAW = BASE / 'data/raw/MAGFiLO_1.0_Kaggle_2026'


def main():
    split = json.load(open(BASE / 'data/split.json'))['val'][:60]
    d = json.load(open(RAW / 'train/MAGFiLO_1.0_Annotations_kaggle2026_train.json'))
    imgs = {str(i['id']): i for i in d['images']}
    want = set(split)
    gt_by = collections.defaultdict(list)
    for a in d['annotations']:
        im = imgs[str(a['image_id'])]
        if im['file_name'] in want:
            gt_by[im['file_name']].append(a)
    ck = torch.load(BASE / 'runs/unet1024/best.pt', map_location='cpu', weights_only=False)
    um = smp.Unet(encoder_name='resnet34', encoder_weights=None, in_channels=3, classes=1)
    um.load_state_dict(ck['model']); um.eval().cuda()
    ym = YOLO(str(BASE / 'runs/s1280/weights/best.pt'))
    rows = []
    for f in split:
        gray = cv2.imread(str(RAW / 'train/train_images' / f), cv2.IMREAD_GRAYSCALE)
        H, W = gray.shape
        small = cv2.resize(gray, (1024, 1024), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(small.astype(np.float32)/255.0)[None,None].repeat(1,3,1,1).cuda()
        with torch.no_grad():
            p = torch.sigmoid(um(x))[0,0].cpu().numpy()
        prob = cv2.resize(p, (W, H), interpolation=cv2.INTER_LINEAR)
        r = ym.predict(str(RAW / 'train/train_images' / f), imgsz=1280, conf=0.15,
                       iou=0.7, max_det=100, retina_masks=True, verbose=False)[0]
        pmasks = [] if r.masks is None else [(r.masks.data.cpu().numpy()[j] > 0.5).astype(np.uint8)
                                             for j in range(r.masks.data.shape[0])]
        g = []
        for a in gt_by.get(f, []):
            m = np.zeros((H, W), np.uint8)
            for poly in a['segmentation']:
                cv2.fillPoly(m, [np.round(np.asarray(poly).reshape(-1,2)).astype(np.int32)], 1)
            g.append((m, a['area'], a['bbox']))
        # 日面半径（用 Otsu 估计圆盘）
        _, disk = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        n_, lab, st, _ = cv2.connectedComponentsWithStats(disk, 8)
        if n_ < 2: continue
        big = 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])
        ys, xs = np.nonzero(lab == big)
        cy, cx, RR = ys.mean(), xs.mean(), max(np.hypot(ys-ys.mean(), xs-xs.mean()).max(), 1)
        for m, area, bbox in g:
            hit = False; best = 0
            for pm in pmasks:
                inter = np.logical_and(m, pm).sum()
                iou = inter / max(m.sum() + pm.sum() - inter, 1)
                best = max(best, iou)
                if iou > 0.5: hit = True; break
            yy, xx = np.nonzero(m)
            if len(yy) == 0: continue
            rpos = float(np.hypot(yy.mean()-cy, xx.mean()-cx) / RR)
            contrast = float(gray[m>0].mean()) - float(gray.max()*0.0)
            rows.append(dict(hit=hit, area=area, r=rpos, contrast=contrast, best=best))
    # summarise
    hit = np.array([r['hit'] for r in rows]); area = np.array([r['area'] for r in rows])
    rr = np.array([r['r'] for r in rows])
    print('instances', len(rows), 'recalled', hit.sum(), f'{100*hit.mean():.1f}%')
    for lo, hi in ((0, 400), (400, 1000), (1000, 3000), (3000, 10**9)):
        m = (area >= lo) & (area < hi)
        if m.sum(): print(f'area [{lo},{hi}): n={m.sum()} recall={100*hit[m].mean():.0f}%')
    for lo, hi in ((0, .3), (.3, .5), (.5, .7), (.7, 1.01)):
        m = (rr >= lo) & (rr < hi)
        if m.sum(): print(f'r/R [{lo},{hi}): n={m.sum()} recall={100*hit[m].mean():.0f}%')
    # false positives
    print('mean best-IoU for missed:', np.mean([r['best'] for r in rows if not r['hit']]))


if __name__ == '__main__':
    main()
