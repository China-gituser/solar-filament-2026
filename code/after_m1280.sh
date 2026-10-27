#!/bin/bash
# 依次评测 m1280（单模型 / 加 refine / 与 s1280 集成）
set -e
cd /data/research/sota_method_2026-10-24/solar_filament
echo "=== m1280 refine (50 imgs) ==="
python3 -u code/fuse_yolo_unet.py --yolo runs/m1280/weights/best.pt --n 50 --conf 0.5 \
  --modes refine --uthr-list 0.85,0.9
echo "=== m1280 conf scan ==="
python3 -u code/fuse_yolo_unet.py --yolo runs/m1280/weights/best.pt --n 50 \
  --conf-list 0.35,0.45,0.55 --conf-low 0.15 --modes refine --uthr-list 0.85
echo "=== ensemble s1280 + m1280 ==="
python3 -u code/fuse_yolo_unet.py --yolo runs/s1280/weights/best.pt runs/m1280/weights/best.pt \
  --n 50 --conf 0.5 --modes refine --uthr-list 0.85
echo "=== done ==="
