#!/usr/bin/env python3
"""Panoptic Quality for the Solar Filament Segmentation Challenge 2026.

Metric semantics mirrored from the organisers' self-evaluation notebook:
  * one ground-truth image record per annotator pass; predictions are grouped by
    the physical image name embedded in `filament_id`
  * every (gt, pred) pair with IoU > 0.5 is a true-positive pair -- matching is
    NOT restricted to a one-to-one assignment
  * FP = predictions with no match, FN = ground-truth instances with no match
  * PQ = sum(IoU over TP pairs) / (TP + 0.5*FP + 0.5*FN)
Per-image terms are also returned so they can be pooled or averaged.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils

H = W = 2048
IOU_THR = 0.5


def _counts_text(rle):
    c = rle['counts']
    return c.decode('ascii') if isinstance(c, bytes) else str(c)


def polygon_rle(segmentation, height=H, width=W):
    rle = mask_utils.merge(mask_utils.frPyObjects(segmentation, height, width))
    return {'size': [height, width], 'counts': _counts_text(rle)}


def decode_rles(rles, height=H, width=W):
    if len(rles) == 0:
        return np.zeros((0, height, width), np.uint8)
    dec = mask_utils.decode([{'size': [height, width], 'counts': r} for r in rles])
    if dec.ndim == 2:
        dec = dec[:, :, None]
    return np.moveaxis(dec, 2, 0)


def load_ground_truth(path, physical_of=None):
    """Return {physical_id: (annotator_image_id, [rle_text, ...])}.

    physical_of: optional dict image_id -> physical stem. By default the stem of
    file_name is used.
    """
    payload = json.loads(Path(path).read_text())
    images = {str(i['id']): i for i in payload['images']}
    grouped = defaultdict(list)
    for a in payload['annotations']:
        im = images[str(a['image_id'])]
        grouped[str(a['image_id'])].append(
            polygon_rle(a['segmentation'], im['height'], im['width']))
    out = {}
    for iid, rec in images.items():
        phys = (physical_of.get(iid) if physical_of else None) or Path(rec['file_name']).stem
        out[iid] = (phys, [r['counts'] for r in grouped.get(iid, [])])
    return out


def load_predictions(path):
    """Return {physical_id: [rle_text, ...]} from a submission CSV."""
    grouped = defaultdict(list)
    with open(path, newline='', encoding='utf-8-sig') as fh:
        rd = csv.DictReader(fh)
        if rd.fieldnames != ['filament_id', 'segmentation_rle']:
            raise ValueError(f'bad header: {rd.fieldnames}')
        for row in rd:
            fid = (row['filament_id'] or '').strip()
            if '_' not in fid:
                raise ValueError(f'bad filament_id {fid!r}')
            grouped[fid.split('_', 1)[0]].append((row['segmentation_rle'] or '').strip())
    return grouped


def score(gt, pred, iou_thr=IOU_THR):
    """Score one (gt list, pred list) pair of RLE text lists."""
    g = decode_rles(gt)
    p = decode_rles(pred)
    if len(g) == 0 and len(p) == 0:
        return dict(tp_pairs=0, fp=0, fn=0, iou_sum=0.0, one_to_many=0,
                    many_to_one=0, n_gt=0, n_pred=0)
    if len(g) == 0:
        return dict(tp_pairs=0, fp=len(p), fn=0, iou_sum=0.0, one_to_many=0,
                    many_to_one=0, n_gt=0, n_pred=len(p))
    if len(p) == 0:
        return dict(tp_pairs=0, fp=0, fn=len(g), iou_sum=0.0, one_to_many=0,
                    many_to_one=0, n_gt=len(g), n_pred=0)
    gf = g.reshape(len(g), -1).astype(np.float32)
    pf = p.reshape(len(p), -1).astype(np.float32)
    inter = gf @ pf.T
    ga = gf.sum(1, keepdims=True)
    pa = pf.sum(1, keepdims=True).T
    union = ga + pa - inter
    iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    hits = iou > iou_thr
    return dict(
        tp_pairs=int(hits.sum()),
        fp=int((hits.sum(0) == 0).sum()),
        fn=int((hits.sum(1) == 0).sum()),
        iou_sum=float(iou[hits].sum()),
        one_to_many=int((hits.sum(1) > 1).sum()),
        many_to_one=int((hits.sum(0) > 1).sum()),
        n_gt=len(g), n_pred=len(p))


def evaluate(gt_path, pred_csv, iou_thr=IOU_THR, per_image=False):
    gt = load_ground_truth(gt_path)
    pred = load_predictions(pred_csv)
    agg = defaultdict(float)
    rows = []
    for iid, (phys, gt_rles) in gt.items():
        pr = pred.get(phys, [])
        s = score(gt_rles, pr, iou_thr)
        for k, v in s.items():
            agg[k] += v
        pq = (s['iou_sum'] / (s['tp_pairs'] + 0.5 * s['fp'] + 0.5 * s['fn'])
              if (s['tp_pairs'] + 0.5 * s['fp'] + 0.5 * s['fn']) > 0 else float('nan'))
        s.update(image_id=iid, physical_id=phys, pq=pq)
        rows.append(s)
    den = agg['tp_pairs'] + 0.5 * agg['fp'] + 0.5 * agg['fn']
    agg['pq'] = agg['iou_sum'] / den if den else 0.0
    agg['sq'] = agg['iou_sum'] / agg['tp_pairs'] if agg['tp_pairs'] else 0.0
    agg['rq'] = agg['tp_pairs'] / den if den else 0.0
    agg['n_images'] = len(gt)
    agg['n_physical_images_with_preds'] = len(pred)
    if per_image:
        agg['per_image'] = rows
    return dict(agg)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt', required=True)
    ap.add_argument('--pred', required=True)
    ap.add_argument('--iou', type=float, default=IOU_THR)
    ap.add_argument('--per-image', action='store_true')
    a = ap.parse_args()
    r = evaluate(a.gt, a.pred, a.iou, a.per_image)
    print(json.dumps({k: v for k, v in r.items() if k != 'per_image'}, indent=1))
