# Solar Filament Segmentation Challenge 2026 — solution

Instance segmentation of solar filaments in 2048×2048 GONG H-alpha images,
scored with the organisers' Panoptic Quality (IoU>0.5 matching, pooled over
annotator passes, FP and FN weighted 0.5 each).

## Method

1. **YOLO11-seg proposals** trained on the union of all annotator passes of each
   physical image (`yolo/`), 1280 px, ~6k instances.
2. **U-Net semantic prior** (resnet34, 1024 px, same union targets) predicts a
   full-resolution probability field.
3. **Proposal refinement**: each YOLO instance is re-thresholded against the
   U-Net field inside a dilated region of interest (`code/predict_refine.py`).
   This improves both mask IoU and matching recall (+0.033 PQ on the local val).
4. **Mutual exclusion**: Kaggle rejects overlapping masks, so instances are
   assigned greedily by confidence.

## Metric implementation

`code/pq_official.py` reproduces the organisers' self-evaluation notebook:
ground truth is grouped by annotator-image (`<annotator>-<image>_<n>`),
predictions by physical image, every pair with IoU>0.5 is a true positive
(not one-to-one), and PQ is pooled globally.

## Reproduce

```bash
# data (Kaggle CLI, competition rules must be accepted)
kaggle competitions download -c filament-segmentation-2026 -p data/raw
unzip data/raw/filament-segmentation-2026.zip -d data/raw

python3 code/prepare_data.py          # YOLO dataset from the COCO json
python3 code/merge_annotators.py      # merge duplicate instances across annotators
python3 code/train_yolo.py --imgsz 1280 --epochs 100 --name s1280
python3 code/train_unet.py --size 1024 --epochs 60 --name unet1024
python3 code/predict_refine.py --yolo runs/s1280/weights/best.pt --conf 0.5 --uthr 0.85
```
