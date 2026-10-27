# Solar Filament Instance Segmentation under Multi-Annotator Panoptic Evaluation
### IEEE BigData Cup 2026 — technical report

## 1. Task and metric

Given a 2048×2048 GONG H-alpha observation, predict one mask per solar
filament. The organiser's metric is panoptic quality with IoU>0.5 matching:

  PQ = Σ IoU(TP) / (|TP| + 0.5|FP| + 0.5|FN)

evaluated **per annotator pass** and pooled. Submissions are single CSV files
(`filament_id,segmentation_rle`), and masks of one image must not overlap.

## 2. What the data actually contains

The 707 training images carry **1,154 annotator passes**: 411 images have one
expert pass, 145 have two, 151 have three; 8,199 instance annotations in total.

Because each pass is scored independently:

* a filament marked by one expert is a TP for that pass and an FP for every
  other pass of the same image;
* FP and FN carry equal weight, so recall cannot be traded for precision;
* two experts scoring each other's labels reach only **PQ 0.320** (SQ 0.631,
  RQ 0.507) — annotation noise is the dominant ceiling.

Using the union of all expert instances as a perfect prediction bounds any
method. On our held-out split this oracle scores **PQ 0.782** (SQ 0.921,
RQ 0.849), but the bound is strongly annotation-count dependent:

| passes/image | images | oracle PQ |
|---|---|---|
| 1 | 66 | 1.000 |
| 2 | 16 | 0.733 |
| 3 | 24 | 0.601 |

Consequently any split-to-split comparison is only meaningful at fixed
annotation-count composition, and the public/private leaderboard gap is
expected when the public half has more multi-pass images.

## 3. Method

**Union targets.** Duplicate instances across passes of one physical image are
merged (mask IoU>0.6, 6,941 → 5,810 instances), so the detector learns to cover
every filament any expert considered real. Training on single-pass labels
instead raises YOLO's own mask mAP50 from 0.52 to 0.68 yet *lowers* PQ
(0.370 vs 0.366): it buys precision with recall, and the metric weights both.

**Detector.** YOLO11s-seg at 1280 px, 100 epochs, mosaic + scale augmentation,
no colour jitter (the input is greyscale).

**Semantic prior and refinement.** A U-Net (resnet34, 1024 px, union targets)
predicts a full-resolution probability field. Each detector proposal is
re-thresholded against that field inside a dilated ROI (6 px); components below
200 px fall back to the original proposal. This moves boundaries toward the
image evidence without inventing or deleting instances:

| stage | PQ | SQ | RQ |
|---|---|---|---|
| YOLO only | 0.366 | 0.649 | 0.564 |
| + U-Net refine (uthr 0.85) | **0.399** | 0.670 | 0.595 |

**Mutual exclusion.** The platform rejects overlapping masks, so instances are
assigned greedily by confidence; this constraint alone costs ≈0.018 PQ and is
applied in every local number reported here.

## 4. Negative results

| idea | PQ | why it fails |
|---|---|---|
| single-annotator labels | 0.370 | recall loss outweighs better masks |
| U-Net components merged into proposals | 0.348 | added instances are mostly FP |
| flip TTA | 0.371 | filaments are not mirror-symmetric |
| 1536 inference from a 1280 model | 0.391 | resolution mismatch |
| YOLO11m instead of s | 0.396 | larger mAP, same PQ |
| s+m ensemble | 0.373 | predictors too correlated, FP doubles |
| tiled 1024 inference | 0.320 | targets no longer at training scale |
| proposal re-ranking (8 features) | no held-out gain | overfits the fitting half |
| 200-epoch training | 0.393 | early-stops, no PQ gain |
| contrast channels [I, σ64, σ12] | 0.387 | detection metrics rise, PQ does not |

## 5. Error analysis and limits

Splitting validation by annotation count locates the bottleneck:

| passes | oracle | measured | SQ | RQ |
|---|---|---|---|---|
| 1 (66) | 1.000 | 0.377 | 0.672 | 0.561 |
| 2 (16) | 0.733 | 0.370 | 0.665 | 0.556 |
| 3 (24) | 0.601 | 0.437 | 0.670 | 0.652 |

Mask quality is flat (SQ ≈ 0.67); single-pass images lose 53% of instances to
missed detections. Missed instances have a mean best-IoU of 0.273, i.e. they are
found but their boundary falls just short of the 0.5 match threshold. Recall is
flat across radial position (72–76%), so limb darkening is not the cause.

## 6. Conclusion

Local validation PQ is **0.399**, public leaderboard **0.340**. Under the
organiser's multi-annotator protocol the achievable ceiling on this split is
0.782, and the binding constraints are (a) 707 training images, (b) thin
low-contrast targets, and (c) FP/FN symmetry that punishes both over- and
under-prediction. U-Net-guided proposal refinement is the one component that
transfers to the leaderboard (+0.05).
