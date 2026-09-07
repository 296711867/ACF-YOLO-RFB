# Phase 2 官方 P2 640/960 分辨率实验准备报告

> 工作包：`P2-RESOLUTION-PREP`  
> 日期：2026-07-29（Asia/Shanghai）  
> 结论：P2-960、batch 16 工程 smoke 通过；640/960 公平配对配置已冻结，可以进入 50 epoch 筛选。

## 1. 实验边界

- 唯一模型为已验收的官方 YOLO26n-P2。
- 使用原生 CIoU/L1、分类和标签分配器，`sa_nwd=false`。
- 不包含 SDD、DWF、注意力或新增数据增强。
- 640 和 960 从同一原始 `yolo26n.pt` 初始化。
- 两组物理 batch 均为 16、`nbs=64`，梯度累积口径相同；配置差异严格只有 `imgsz` 和 run name。

## 2. P2-960 固定 smoke

配置：`configs/train/p2-resolution-smoke-960.yaml`。

- 固定 BDD100K 400 train / 100 val、960、batch 16、seed 0、GPU0。
- 完成 3/3 epoch，无 early stopping、fitness 自动恢复、OOM、NaN 或 Inf。
- 最高训练日志显存约 14.7 GB，RTX 3090 有足够安全余量。
- `results.csv` 3 行全部有限；best/last、曲线、最终验证与元数据完整。
- best checkpoint 重载通过，检测 stride 为 `[4, 8, 16, 32]`。

run：`runs/detect/phase2-smoke/bdd100k-yolo26n-p2-native-960-e3-b16-seed0/`。

## 3. 配对筛选

| 项目 | 640 control | 960 candidate |
|---|---|---|
| 配置 | `p2-resolution-screen-640.yaml` | `p2-resolution-screen-960.yaml` |
| imgsz | 640 | 960 |
| batch / nbs | 16 / 64 | 16 / 64 |
| epochs / seed | 50 / 0 | 50 / 0 |
| 模型/损失 | 官方 P2 / 原生 | 官方 P2 / 原生 |

完成后必须对 best checkpoint 进行同分辨率独立 val 和 COCO 尺寸分组评估，并报告训练时间、峰值显存、参数/GFLOPs和验证推理速度。

## 4. 证据哈希

| 文件 | SHA256 |
|---|---|
| 960 smoke 配置 | `b5d4204e72466c498367848f479f189b0bba16f86ead6297433e0b70927242c7` |
| 640 screen 配置 | `64c86ead51afe7c742a0a09e67857e99f5c4214b31a19c93e3d8d1952bbc22d9` |
| 960 screen 配置 | `a741d76752a9809a08c488641349993605cd86bd777ca210146f64b573e3da5e` |
| smoke `results.csv` | `baeac5407619307be956a53ed74784db13072b665f48ff53690371c2e1482b63` |
| smoke best | `eeda8be393b4a798f58eb8bd2450923df5fa0815b5c7d17caffe333d8dad16ab` |
| smoke last | `164412a8b2444d8bbffff8437b1af796b381aa65531622392a831dbf56edcf0e` |
