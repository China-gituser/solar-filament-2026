#!/usr/bin/env python3
"""Cross-annotator PQ: score one expert's labels against another expert's labels
for the same physical GONG image. Gives a human-consistency ceiling."""
import json, sys, collections, itertools
import numpy as np
from pycocotools import mask as mu

P='/data/research/sota_method_2026-10-24/solar_filament/data/magfilo_2024_v1.0.json'

def rle(seg, h, w):
    return mu.merge(mu.frPyObjects(seg, h, w))

def dec(rles, h=2048, w=2048):
    if not rles: return np.zeros((0,h,w), np.uint8)
    d=mu.decode(rles)
    if d.ndim==2: d=d[:,:,None]
    return np.moveaxis(d,2,0)

def pq_pair(g_rles, p_rles, thr=0.5):
    g=dec(g_rles); p=dec(p_rles)
    if len(g)==0 or len(p)==0:
        return dict(tp=0, fp=len(p), fn=len(g), iou_sum=0.0)
    gf=g.reshape(len(g),-1).astype(np.float32); pf=p.reshape(len(p),-1).astype(np.float32)
    inter=gf@pf.T; ga=gf.sum(1,keepdims=True); pa=pf.sum(1,keepdims=True).T
    u=ga+pa-inter
    iou=np.divide(inter,u,out=np.zeros_like(inter),where=u>0)
    hit=iou>thr
    return dict(tp=int(hit.sum()), fp=int((hit.sum(0)==0).sum()), fn=int((hit.sum(1)==0).sum()),
                iou_sum=float(iou[hit].sum()))

def main():
    d=json.load(open(P))
    imgs={str(i['id']):i for i in d['images']}
    anns=collections.defaultdict(list)
    for a in d['annotations']: anns[str(a['image_id'])].append(a)
    by_file=collections.defaultdict(list)
    for iid,im in imgs.items(): by_file[im['file_name']].append(iid)
    multi=[(f,v) for f,v in by_file.items() if len(v)>1]
    print('physical images with >1 annotator record:', len(multi))
    rng=np.random.default_rng(0)
    sel=rng.permutation(len(multi))[:120]
    T=F=N=0; IS=0.0; per=[]
    for k in sel:
        f,ids=multi[k]
        for a_id,b_id in itertools.combinations(ids,2):
            h,w=imgs[a_id]['height'],imgs[a_id]['width']
            ga=[rle(x['segmentation'],h,w) for x in anns[a_id]]
            gb=[rle(x['segmentation'],h,w) for x in anns[b_id]]
            for g,p in ((ga,gb),(gb,ga)):
                r=pq_pair(g,p); T+=r['tp']; F+=r['fp']; N+=r['fn']; IS+=r['iou_sum']
    den=T+0.5*F+0.5*N
    print(f'pairs scored, TP={T} FP={F} FN={N}')
    print('cross-annotator PQ = %.4f'%(IS/den if den else 0))
    print('SQ = %.4f  RQ = %.4f'%(IS/T if T else 0, T/den if den else 0))

if __name__=='__main__': main()
