#!/usr/bin/env python3
"""Panoptic Quality exactly as the organisers' self-evaluation notebook computes it.

From the official notebook (kernel `azimahmadzadeh/self-evaluation-notebook`):

    gt_annotator_image_ids = gt_df["filament_id"].str.split("_", n=1).str[0]
    pred_image_ids         = pred_df["filament_id"].str.split("_", n=1).str[0]
    for each unique gt annotator-image:
        image_id = annotator_image.split("-", maxsplit=1)[1]
        gt   = gt rows whose annotator-image == annotator_image
        pred = pred rows whose image_id == image_id
        IoU matrix; hit = IoU > 0.5
        TP pairs = every hit; FP = predictions with no hit; FN = gt with no hit
    PQ = sum(IoU over TP pairs) / (TP + 0.5 FP + 0.5 FN)        # pooled globally

So a ground-truth filament_id is "<annotator_id>-<image_id>_<n>" and a prediction
filament_id is "<image_id>_<n>". Predictions are reused against every annotator
pass of the same physical image.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from pycocotools import mask as mu

H = W = 2048
THR = 0.5


def _counts(rle):
    c = rle['counts']
    return c.decode('ascii') if isinstance(c, bytes) else str(c)


def rasterize(segmentation, h=H, w=W):
    return _counts(mu.merge(mu.frPyObjects(segmentation, h, w)))


def decode(rles, h=H, w=W):
    if not rles:
        return np.zeros((0, h, w), np.float32)
    d = mu.decode([{'size': [h, w], 'counts': r} for r in rles])
    if d.ndim == 2:
        d = d[:, :, None]
    return np.moveaxis(d, 2, 0).astype(np.float32)


def gt_from_coco(json_path):
    """Return {annotator_image: [rle_text, ...]} from a COCO json."""
    payload = json.loads(Path(json_path).read_text())
    imgs = {str(i['id']): i for i in payload['images']}
    out = defaultdict(list)
    for a in payload['annotations']:
        im = imgs[str(a['image_id'])]
        out[str(a['image_id'])].append(
            rasterize(a['segmentation'], im['height'], im['width']))
    for iid in imgs:
        out.setdefault(str(iid), [])
    return dict(out)


def gt_from_csv(csv_path):
    out = defaultdict(list)
    with open(csv_path, newline='', encoding='utf-8-sig') as fh:
        for row in csv.DictReader(fh):
            fid = (row['filament_id'] or '').strip()
            out[fid.split('_', 1)[0]].append((row['segmentation_rle'] or '').strip())
    return dict(out)


def pred_from_csv(csv_path):
    out = defaultdict(list)
    with open(csv_path, newline='', encoding='utf-8-sig') as fh:
        rd = csv.DictReader(fh)
        if rd.fieldnames != ['filament_id', 'segmentation_rle']:
            raise ValueError(f'bad header {rd.fieldnames}')
        for row in rd:
            fid = (row['filament_id'] or '').strip()
            out[fid.split('_', 1)[0]].append((row['segmentation_rle'] or '').strip())
    return dict(out)


def _rle_dicts(rles, h=H, w=W):
    return [{'size': [h, w], 'counts': (r.encode('ascii') if isinstance(r, str) else r)}
            for r in rles]


def iou_matrix(gt, pred, h=H, w=W):
    """IoU via pycocotools on compressed RLE -- never materialises HxW masks."""
    if not gt or not pred:
        return np.zeros((len(gt), len(pred)), np.float64)
    return np.asarray(mu.iou(_rle_dicts(gt, h, w), _rle_dicts(pred, h, w),
                             [0] * len(pred)), np.float64)


def evaluate(gt_map, pred_map, thr=THR, per_entry=False):
    """gt_map: {annotator_image: rle list}; pred_map: {image_id: rle list}."""
    tp_iou, fp, fn, n_pair = [], 0, 0, 0
    rows = []
    for ann_img, gt_rles in gt_map.items():
        if '-' not in ann_img:
            # a ground-truth id without an annotator prefix still works: the whole
            # id is treated as the physical image id
            image_id = ann_img
        else:
            image_id = ann_img.split('-', 1)[1]
        pr = pred_map.get(image_id, [])
        if len(gt_rles) == 0:
            fp += len(pr)
            rows.append(dict(annotator_image=ann_img, n_gt=0, n_pred=len(pr),
                             tp=0, iou_sum=0.0, fp=len(pr), fn=0))
            continue
        if len(pr) == 0:
            fn += len(gt_rles)
            rows.append(dict(annotator_image=ann_img, n_gt=len(gt_rles), n_pred=0,
                             tp=0, iou_sum=0.0, fp=0, fn=len(gt_rles)))
            continue
        iou = iou_matrix(gt_rles, pr)
        hit = iou > thr
        t = float(iou[hit].sum())
        tp_iou.extend(iou[hit].tolist())
        f = int((hit.sum(0) == 0).sum())
        n = int((hit.sum(1) == 0).sum())
        fp += f
        fn += n
        n_pair += int(hit.sum())
        rows.append(dict(annotator_image=ann_img, n_gt=len(gt_rles), n_pred=len(pr),
                         tp=int(hit.sum()), iou_sum=t, fp=f, fn=n))
    tp = len(tp_iou)
    den = tp + 0.5 * fp + 0.5 * fn
    res = dict(pq=(sum(tp_iou) / den if den else 0.0),
               sq=(sum(tp_iou) / tp if tp else 0.0),
               rq=(tp / den if den else 0.0),
               tp=tp, fp=fp, fn=fn, iou_sum=float(sum(tp_iou)),
               n_gt_entries=len(gt_map))
    if per_entry:
        res['entries'] = rows
    return res


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt-json')
    ap.add_argument('--gt-csv')
    ap.add_argument('--pred', required=True)
    ap.add_argument('--thr', type=float, default=THR)
    ap.add_argument('--lazy', action='store_true')
    a = ap.parse_args()
    gt = gt_from_coco(a.gt_json) if a.gt_json else gt_from_csv(a.gt_csv)
    pr = pred_from_csv(a.pred)
    r = evaluate(gt, pr, a.thr)
    print(json.dumps({k: v for k, v in r.items() if k != 'entries'}, indent=1))
