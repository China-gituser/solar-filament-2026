# Solar Filament Segmentation Challenge 2026 —— 项目状态

## 基本情况
- Kaggle 比赛 id 143678，IEEE BigData Cup 2026，赞助 NSO/NSF，奖金 $3,000
- **截止 2026-11-15 06:00 UTC**（规则页写 11-15 23:59 CDT；API deadline 为 UTC 06:00）
- 每天 5 次提交；最终需 4 页技术报告 + 公开 Git 仓库（含 requirements.txt）+ 端到端 notebook
- 指标 **panoptic_quality**（官方 self-eval notebook 语义，见下）
- 榜单：1,202 队；top1 0.70 / top2 0.64 / top50 约 0.56
- 数据：707 张 train（2048² 灰度 JPEG）+ 180 张 test；全部来自 MAGFiLO v1.0

## 数据结构（已核实）
- 训练标注 JSON：`MAGFiLO_1.0_Annotations_kaggle2026_train.json`
- 707 个物理图像 → **1,154 个 annotator-image 记录**（411 单标注 / 145 双标注 / 151 三标注）
- 8,199 个实例标注，每图中位 7 个；类别 1/2/3（磁手性）
- image_id 形如 `040301-20140609195854Bh`（`<annotator_id>-<image_id>`）
- MAGFiLO 全量 958 张唯一图 / 1,593 记录，比赛 train/test 无交叉

## 官方 PQ 精确语义（已从 kernel 源码核实）
- GT 的 filament_id = `<annotator_id>-<image_id>_<n>`；预测的 = `<image_id>_<n>`
- 对**每个 annotator-image** 单独取该物理图像的全部预测，算 IoU 矩阵
- IoU > 0.5 的每个 pair 都算 TP（**非一对一匹配**）
- FP = 无任何匹配的预测；FN = 无任何匹配的 GT
- **全局池化**：PQ = ΣIoU(TP) / (|TP| + 0.5|FP| + 0.5|FN)

## 关键实验与结论
1. **跨标注者一致性**：随机 120 张多标注图，A 的标注当预测、B 的当 GT：
   **PQ = 0.320（SQ 0.631 / RQ 0.507）**。标注噪声很大。
2. **完美并集上限**（val，106 图 / 170 passes）：
   `PQ=0.7823, SQ=0.9211, RQ=0.8493, TP=1254, FP=438, FN=7`
   —— 以所有 annotator 实例的跨标注者 IoU>0.5 聚类并集作为完美预测。
3. **共识过滤反而更差**：只保留被 ≥2 个 annotator 标注的实例 → PQ 0.404；
   ≥3 → 0.215。原因：并集约 80% 的实例只出现在一个 annotator 记录里（多数来自单标注图）。
4. 结论：**最优策略是高召回 + 高精度 mask，预测所有真实细丝**，而不是模仿单套标签或做共识化。
   top1 的 0.70 约为并集上限的 90%。

## 已建资产
- `code/pq_official.py`：官方语义 PQ 评测器（RLE 直接算 IoU，支持多 annotator 池化）
- `code/prepare_data.py`：COCO → YOLO-seg，按物理图分组 601/106 划分（`data/split.json`）
- `code/merge_annotators.py`：跨标注者重复实例合并（IoU>0.6），训练标签 6,941 → 5,810
- `code/eval_yolo_val.py`：YOLO 预测 → 官方 PQ（val）
- `code/ceiling_union.py` / `code/oracle_consensus.py`：上限与策略 oracle 实验
- `data/raw/MAGFiLO_1.0_Kaggle_2026/`：707 train + 180 test

## 训练
- `runs/s1280`：YOLO11s-seg，imgsz 1280，batch 8，100 epochs，mosaic 1.0，~15 s/epoch
- epoch 3 时 mask mAP50=0.35（YOLO 自有指标，标签为合并并集）

## 待办
1. 基线训练完成 → 官方 PQ（含 conf 阈值扫描）
2. 提高召回：多尺度/TTA、tile 推理、更强 backbone
3. mask 精度：retina masks、更高 imgsz
4. 提交格式校验与首次线上提交

---

## 实验日志（2026-10-27 下午）

所有 PQ 均为官方语义、含 Kaggle 强制的**同图 mask 互斥**约束。

| 实验 | 设置 | 本地 val PQ | 备注 |
|---|---|---|---|
| s1280 | YOLO11s-seg 1280，并集标签，100 ep | 0.366（conf 0.50） | 无互斥时 0.384 |
| single1280 | 同上，单标注者标签 | 0.370（conf 0.40） | YOLO mAP50 0.68 但 PQ 未提升；FN 604 vs 551 |
| unet1024 | U-Net resnet34 1024，并集语义 | 0.382（thr 0.90） | 连通域实例化 |
| **refine** | YOLO 提案 + U-Net 概率重阈值（uthr 0.85） | **0.399** | +0.033；SQ 0.670 / RQ 0.595 |
| merge | YOLO 提案 ∪ U-Net 连通域 | 0.348 | 失败：补的实例多为 FP（FP 439 vs 185） |
| 1536 推理 | s1280 权重，imgsz 1536 | 0.391 | 与训练分辨率不一致，下降 |
| 线上提交 1 | s1280 conf 0.40 互斥 | **0.290**（public） | 本地同口径 0.362 |

关键结论：
1. **互斥约束损失约 0.018 PQ**，必须在本地区分。
2. 标签用并集还是单标注者，在官方 PQ 上几乎无差别（0.366 vs 0.370），
   尽管 YOLO 自带 mAP50 相差 0.16 —— **不能看 mAP 选模型**。
3. **U-Net 精修 YOLO 提案是目前唯一稳定增益**（+0.033）。
4. U-Net 连通域单独用与 YOLO 相当，但补充实例会引入大量 FP。
5. 本地 val 比线上 public 高约 0.07（0.366 vs 0.290），原因待查
   （val/test 人群差异最可能；两边都是随机划分）。

## 并集上限按标注者数拆分（val）

| 每图标注者数 | 图数 | oracle PQ | SQ | RQ |
|---|---|---|---|---|
| 1 | 66 | **1.0000** | 1.000 | 1.000 |
| 2 | 16 | 0.7328 | 0.906 | 0.809 |
| 3 | 24 | 0.6009 | 0.837 | 0.718 |
| 全部 | 106 | 0.7823 | 0.921 | 0.849 |

含义：
- 单标注图只要检测齐全就能拿满分；难度全部来自多标注图。
- 若 test 的三标注比例高于 val，线上分数会系统性低于本地（这与实测 0.34 vs 0.399 一致）。
- 0.70 的线上分要求在三标注图（上限 0.60）上几乎满分，或在单标注图上
  接近完美——前者不可能，说明 public test 的单标注占比高于 val，或
  榜单前列在别的维度上有结构性优势。

## 分组评测（refine 流程，conf 0.50 / uthr 0.85）

| 每图标注者数 | 图 | oracle 上限 | 实测 PQ | SQ | RQ | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|
| 1 | 66 | 1.000 | **0.377** | 0.672 | 0.561 | 263 | 119 | 293 |
| 2 | 16 | 0.733 | 0.370 | 0.665 | 0.556 | 111 | 61 | 116 |
| 3 | 24 | 0.601 | 0.437 | 0.670 | 0.652 | 283 | 110 | 192 |

**最大瓶颈在单标注图**：上限 1.0，实测 0.377，漏检 53%（FN 293 / TP 263）。
三标注图反而更接近其上限（0.437 / 0.601 = 73%）。因此提升方向是**召回**，
不是 mask 精度（SQ 在各组都稳定在 0.67）。

## 召回杠杆估算（单标注图，固定 SQ=0.672、FP=119）

| 召回 | TP | FN | PQ |
|---|---|---|---|
| 53%（当前） | 263 | 293 | 0.377 |
| 70% | 389 | 167 | 0.492 |
| 80% | 445 | 111 | 0.534 |
| 90% | 500 | 56 | 0.572 |

⇒ 单标注图上每 10 个百分点的召回约值 0.04–0.05 PQ。召回是当前最大的可控杠杆。

## 第二轮负结果（m1280 之后）

| 实验 | 本地 PQ | 结论 |
|---|---|---|
| m1280（yolo11m, 60 ep）+ refine | 0.3961（50 图） | 与 s1280 的 0.3977 持平；mAP50 高 0.037 但 PQ 不涨 |
| s1280 + m1280 集成 | 0.3725 | 两模型高度相关，合并只增 FP（323 vs 156） |
| 切片推理 1024/stride768 | 0.3200（40 图） | 目标放大后与训练分布不符，FP 513 |
| 提案重排序（LR on 8 特征） | held-out 0.3397 / train-half 0.4001 | 过拟合，无 held-out 增益 |

累计：**8 项改进尝试，仅 U-Net refine 有效（+0.033）**。
当前最佳：本地 0.399，线上 0.340。

评估：瓶颈已从 mask 精度转向检测召回，而召回受限于
(a) 707 张训练图的容量，(b) 细丝本身低对比度。
在规则禁止使用 MAGFiLO 其余部分（含 test GT）的前提下，
可用的外部数据只有无标注 GONG 图像，收益不确定。

## 第三轮：对比度通道（c1280）

输入改为 [灰度, 大尺度局部对比度(σ=64), 小尺度局部对比度(σ=12)]，
其余与 s1280 相同（601/106 同划分，YOLO11s-seg，1280，100 ep）。

- 训练期 mask mAP50 0.53–0.55、recall 0.50–0.55，均高于 s1280（0.516 / 0.499）
- 但 epoch 76 的官方 PQ（+U-Net 精修）为 **0.387**，低于 s1280 的 **0.398**
- 又一次"检测指标涨、PQ 不涨"

累计 10 项改进尝试，仅 U-Net 精修有效。这说明在 707 张训练图的约束下，
YOLO 类检测器的 PQ 已接近其可达上限；进一步提升需要不同的范式或数据。

## 规则核查：自训练不可行

Kaggle Foundational Rules："Submissions may not use or incorporate information from
hand labeling or human prediction of the validation dataset or test data records."
官方比赛页："for inference of ML models, only the H-alpha images provided in the
test directory will be used."

自训练（test 图像 + 模型伪标签重训）虽然伪标签不是人工的，但把 test 图像用于训练，
属于规则灰色地带，且可能被视为 unfair playing practice。**已停止，不采用。**
