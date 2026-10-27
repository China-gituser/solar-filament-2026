#!/usr/bin/env python3
"""Re-rank low-confidence proposals with features that predict whether a proposal
matches any ground-truth instance.

A fixed confidence cut-off trades recall against precision blindly. Instead we
describe each proposal (detector score, size, U-Net support, local contrast)
and fit a logistic model on one half of the validation images, then evaluate on
the other half so the reported number is not fitted.
"""
import argparse, json, sys, collections
from pathlib import Path
import cv2, numpy as np, torch
from ultralytics import YOLO
import segmentation_models_pytorch as smp
from pycocotools import mask as mu

sys.path.insert(0, '/data/research/sota_method_2026-10-24/solar_filament/code')
from pq_official import evaluate, decode
from eval_yolo_val import build_gt

BASE = Path('/data/research/sota_method_2026-10-24/solar_filament')
RAW = BASE / 'data/raw/MAGFiLO_1.0_Kaggle_2026'


def load_unet(ckpt):
    ck = torch.load(ckpt, map_location='cpu', weights_only=False)
    m = smp.Unet(encoder_name='resnet34', encoder_weights=None, in_channels=3, classes=1)
    m.load_state_dict(ck['model']); return m.eval().cuda()


def features(m, conf, prob, gray):
    ys, xs = np.nonzero(m)
    if len(ys) == 0:
        return None
    h, w = m.shape
    area = len(ys)
    bh = ys.max() - ys.min() + 1; bw = xs.max() - xs.min() + 1
    inside = prob[ys, xs]
    # local contrast: mean inside vs a dilated ring
    ker = np.ones((21, 21), np.uint8)
    ring = np.logical_and(cv2.dilate(m, ker), np.logical_not(m))
    ring_mean = float(gray[ring].mean()) if ring.sum() else float(gray.mean())
    return [
        conf,
        np.log1p(area),
        np.log1p(bh * bw),
        max(bh, bw) / max(min(bh, bw), 1),
        float(inside.mean()),
        float(inside.max()),
        float(prob[m > 0].mean()) - float(prob[ring].mean()) if ring.sum() else 0.0,
        float(gray[ys, xs].mean()) - ring_mean,
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--yolo', default=str(BASE / 'runs/s1280/weights/best.pt'))
    ap.add_argument('--unet', default=str(BASE / 'runs/unet1024/best.pt'))
    ap.add_argument('--n', type=int, default=106)
    ap.add_argument('--conf', type=float, default=0.15)
    ap.add_argument('--split-frac', type=float, default=0.5)
    a = ap.parse_args()
    split = json.load(open(BASE / 'data/split.json'))['val'][:a.n]
    gt = build_gt(split)
    ym = YOLO(a.yolo); um = load_unet(a.unet)
    rows = []
    for f in split:
        gray = cv2.imread(str(RAW / 'train/train_images' / f), cv2.IMREAD_GRAYSCALE)
        stem = Path(f).stem
        r = ym.predict(str(RAW / 'train/train_images' / f), imgsz=1280, conf=a.conf,
                       iou=0.7, max_det=100, retina_masks=True, verbose=False)[0]
        small = cv2.resize(gray, (1024, 1024), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(small.astype(np.float32) / 255.0)[None, None].repeat(1, 3, 1, 1).cuda()
        with torch.no_grad():
            p = torch.sigmoid(um(x))[0, 0].cpu().numpy()
        prob = cv2.resize(p, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)
        g = [decode([z])[0].astype(np.uint8) for z in gt.get(stem + '', [])]
        # gt keys are annotator-image ids; collect those whose physical id is stem
        g = [decode([z])[0].astype(np.uint8) for k, v in gt.items()
             if k.split('-', 1)[-1] == stem for z in v]
        if r.masks is None:
            continue
        ms = r.masks.data.cpu().numpy(); cf = r.boxes.conf.cpu().numpy()
        for j in range(ms.shape[0]):
            m = (ms[j] > 0.5).astype(np.uint8)
            fv = features(m, float(cf[j]), prob, gray)
            if fv is None:
                continue
            hit = 0
            for gg in g:
                inter = np.logical_and(m, gg).sum()
                if inter / max(m.sum() + gg.sum() - inter, 1) > 0.5:
                    hit = 1; break
            rows.append(dict(stem=stem, feat=fv, hit=hit, conf=float(cf[j]),
                             rle=mu.encode(np.asfortranarray(m))['counts'].decode()))
    print('proposals', len(rows), 'positives', sum(r['hit'] for r in rows))
    stems = sorted({r['stem'] for r in rows})
    rng = np.random.default_rng(0); rng.shuffle(stems)
    ntr = int(len(stems) * a.split_frac)
    tr, te = set(stems[:ntr]), set(stems[ntr:])
    from sklearn.linear_model import LogisticRegression
    Xtr = np.array([r['feat'] for r in rows if r['stem'] in tr])
    ytr = np.array([r['hit'] for r in rows if r['stem'] in tr])
    clf = LogisticRegression(max_iter=1000, class_weight='balanced').fit(Xtr, ytr)
    def score(r): return clf.decision_function([r['feat']])[0]
    for r in rows:
        r['score'] = score(r)
    for tag, group in (('train-half', tr), ('heldout', te)):
        g = {k: v for k, v in gt.items() if k.split('-', 1)[-1] in group}
        for thr in (0.0, 0.2, 0.4, 0.6):
            sel = collections.defaultdict(list)
            for r in rows:
                if r['stem'] in group and r['score'] >= thr:
                    sel[r['stem']].append((r['rle'], r['score']))
            pm = {}
            for stem, items in sel.items():
                items.sort(key=lambda z: -z[1])
                occ = None; out = []
                for rle, _ in items:
                    m = decode([rle])[0].astype(np.uint8)
                    if occ is not None:
                        m = np.logical_and(m, np.logical_not(occ)).astype(np.uint8)
                    if m.sum() == 0: continue
                    occ = m if occ is None else np.logical_or(occ, m)
                    out.append(mu.encode(np.asfortranarray(m))['counts'].decode())
                pm[stem] = out
            res = evaluate(g, pm)
            print(f'{tag} score>={thr:.1f}: PQ={res["pq"]:.4f} SQ={res["sq"]:.4f} '
                  f'RQ={res["rq"]:.4f} TP={res["tp"]} FP={res["fp"]} FN={res["fn"]}')


if __name__ == '__main__':
    main()
