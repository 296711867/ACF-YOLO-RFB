# Phase 1 YOLO26n 640 正式基线验收报告

> 数据集：BDD100K Detection
> 模型：YOLO26n，原生 P3/P4/P5
> 训练：640、200 epoch、batch 64、seed 0
> 结论：通过
> 完成时间：2026-07-26 23:08:18（Asia/Shanghai）

## 1. 冻结输入与证据

| 项目 | 值 |
|---|---|
| 训练 commit | `87ba6eca2cd5f44c2ac93bfbab2d8435333459d8` |
| 续训 commit | `261085e10f80e51cb060206695de82d9eadaaedb` |
| 配置 | `configs/train/bdd100k-640.yaml` |
| 配置 SHA256 | `755e14d7785d3adacc477088b21fefaac01ef22061c01bc576d11f312246f46c` |
| 数据 YAML SHA256 | `99cd88cd7024f4ac5eb4330172366ff2649d32447754dbb6a84220cd098` |
| 初始权重 SHA256 | `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef` |
| 训练图像 | 70,000 |
| 验证图像/实例 | 10,000 / 185,526 |
| 实际设备 | 物理 GPU0，RTX 3090 24 GB |
| Ultralytics | 8.4.104 |
| PyTorch/CUDA | 2.6.0+cu126 / 12.6 |

训练 run：`runs/detect/phase1/bdd100k-yolo26n-640-e200-seed0/`。

## 2. 训练过程

- 完整完成 200/200 epoch，没有触发 early stopping。
- 有效训练时间约 61,499.59 秒，即 17 小时 5 分。
- epoch 17 后前台进程曾被外部终止；无 OOM、重启、NaN 或 GPU 错误证据。
- `last.pt` 保留完整状态后，通过受控 resume 和显式 `CUDA_VISIBLE_DEVICES=0` 从 epoch 18 续训。
- resume 在 tmux 持久会话完成，history、lock、配置哈希和 GPU 映射证据齐全。
- 200 个 epoch 的全部 CSV loss/指标均为有限值。
- 最佳 epoch 为 181；之后 19 epoch 未刷新最佳，低于 `patience=50`。
- 训练结束后 best/last 已正常 strip optimizer，属于完整结束状态。

最佳训练记录：

| epoch | Precision | Recall | mAP50 | mAP50-95 | train box | train cls | train l1 | val box | val cls | val l1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 181 | 0.68630 | 0.42203 | 0.46424 | 0.25473 | 1.95063 | 1.12638 | 0.00315 | 1.84277 | 1.08067 | 0.00413 |

## 3. 独立验证

使用最终 `best.pt` 对完整 BDD100K val 独立执行 Ultralytics val：

| Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|---:|---:|---:|---:|---:|---:|
| 10,000 | 185,526 | 0.687 | 0.423 | 0.464 | 0.255 |

独立验证目录：`runs/detect/phase1/bdd100k-yolo26n-640-e200-seed0-independent-val/`。

## 4. 尺寸专项评估

协议：COCO 风格 101 点 AP，IoU 0.50:0.95，maxDet 300；small `<32²`、medium `32²–96²`、large `>=96²`。

| 尺寸 | GT instances | AP50-95 | AP50 | AR50-95 | AR50 |
|---|---:|---:|---:|---:|---:|
| all | 185,526 | 0.255413 | 0.464085 | 0.424718 | 0.723184 |
| small | 102,729 | 0.088772 | 0.240914 | 0.237142 | 0.543253 |
| medium | 59,669 | 0.290918 | 0.523915 | 0.494821 | 0.791100 |
| large | 23,128 | 0.508208 | 0.720179 | 0.634142 | 0.843993 |

自定义整体 AP50-95 与 Ultralytics 独立 val 的差异小于 0.001，评估口径一致。small AP50-95 仅为 large 的约 17.5%，后续针对小目标的结构与损失改进具有明确实验依据。

分类别整体与 small AP50-95：

| 类别 | all | small |
|---|---:|---:|
| person | 0.248642 | 0.097514 |
| rider | 0.163052 | 0.022047 |
| car | 0.452437 | 0.180111 |
| truck | 0.405129 | 0.076108 |
| bus | 0.413186 | 0.046062 |
| train | 0.000452 | 0.000000 |
| motor | 0.167239 | 0.036202 |
| bike | 0.180336 | 0.023877 |
| traffic light | 0.216053 | 0.184397 |
| traffic sign | 0.307608 | 0.221399 |

完整指标：`results/baseline/bdd100k-yolo26n-640-size-metrics.json`。逐图原始预测共 10,000 行、约 234 MB，保留在同目录但由 Git 忽略。

## 5. 权重与结果哈希

| 文件 | SHA256 |
|---|---|
| `weights/best.pt` | `0fb7a0716d3ac8b646dae9aa3477b88380178a9bcf4a8094dd970ef96cb59941` |
| `weights/last.pt` | `14abc13fe47d35df4e64b4365bd8588403ebb38fa2ac1ed1d0dc40f60b923775` |
| `results.csv` | `cd9af15831c056340194cbfb809779ce73536c6f1d3a3644ab354a4a322fe1a5` |
| 尺寸指标 JSON | `0157db69c5a7a466faae168433eb2c26213dc78d1c3262920b58319b8a12da60` |
| 原始预测 JSONL | `267e35fc282af225132a19892b3ac62d23169a3d82c5b4df761249d1bf3fcfd8` |

## 6. 验收结论

- [x] 200 epoch 正常完成，无 NaN/Inf。
- [x] args、CSV、曲线、best/last、元数据和 resume history 完整。
- [x] 完整独立 val 通过。
- [x] small/medium/large 专项评估通过并保留原始预测。
- [x] 整体、分类别和尺寸指标均可复查。
- [x] YOLO26n 640 正式基线验收通过。

下一步仅执行官方 YOLO26-P2 独立消融；在 P2 结果完成前不实现 SDD、DWF 或 SA-NWD。
