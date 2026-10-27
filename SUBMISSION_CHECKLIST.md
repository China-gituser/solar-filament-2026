# 最终提交清单（截止 2026-11-15 06:00 UTC）

## 必交（缺一不可）
- [ ] **技术报告 4 页 PDF** —— 用官方 Overleaf 模板
      https://www.overleaf.com/read/vjztpvrnbrdh#43fbbe
- [ ] **公开 Git 仓库**（GitHub/GitLab/Bitbucket，无需额外权限即可访问）
      - [ ] `requirements.txt` 含全部包及版本
      - [ ] `solution.ipynb` 端到端复现（预处理→推理→提交文件）
      - [ ] 模型权重**放在仓库内**，notebook 直接加载，不重新训练
- [ ] **Google Form** 提交上述材料
      https://docs.google.com/forms/d/e/1FAIpQLSfyrM4dhc8QzxR90PtF6fb018KPf0S0Cbf5hPqMuUognmzQsA/viewform

## 提交策略
- 每天 5 次提交；最终最多选 **2 个** submission 为最终评分
- 11-15 前选定（规则写明需要提交表单）
- public LB 只覆盖 50% test，最终用另一 50%

## 代码状态
- [x] `code/pq_official.py` 官方语义 PQ 评测
- [x] `code/prepare_data.py` / `merge_annotators.py` 数据准备
- [x] `code/train_yolo.py` / `train_unet.py` 训练
- [x] `code/predict_refine.py` 推理（YOLO 提案 + U-Net 精修 + 互斥）
- [x] `README.md` / `requirements.txt`
- [ ] `solution.ipynb`（待写，把上述流程串成 notebook）

## 风险
- 权重文件大小：yolo11s-seg 20 MB + U-Net 90 MB，可放 Git LFS 或直接提交
- notebook 需在无网环境跑（Kaggle 评分环境默认无网）
- 提交 CSV 的 mask 必须互斥（已验证）
