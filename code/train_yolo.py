#!/usr/bin/env python3
import argparse
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument('--model', default='yolo11s-seg.pt')
ap.add_argument('--data', default='/data/research/sota_method_2026-10-24/solar_filament/yolo/data.yaml')
ap.add_argument('--imgsz', type=int, default=1280)
ap.add_argument('--epochs', type=int, default=100)
ap.add_argument('--batch', type=int, default=8)
ap.add_argument('--workers', type=int, default=8)
ap.add_argument('--name', default='s1280')
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--mosaic', type=float, default=1.0)
ap.add_argument('--close-mosaic', type=int, default=10)
ap.add_argument('--scale', type=float, default=0.5)
ap.add_argument('--lr0', type=float, default=0.01)
ap.add_argument('--patience', type=int, default=40)
ap.add_argument('--resume', action='store_true')
a = ap.parse_args()
m = YOLO(a.model)
m.train(data=a.data, imgsz=a.imgsz, epochs=a.epochs, batch=a.batch, workers=a.workers,
        device=0, project='/data/research/sota_method_2026-10-24/solar_filament/runs',
        name=a.name, seed=a.seed, exist_ok=True, plots=False, verbose=False,
        mosaic=a.mosaic, close_mosaic=a.close_mosaic, scale=a.scale, lr0=a.lr0,
        patience=a.patience, cache=False)
