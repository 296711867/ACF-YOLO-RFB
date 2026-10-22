# BDD100K 时段（timeofday）切片评估——首批证据

> 生成日期：2026-09-28（Asia/Shanghai）
> 性质：对冻结 `*.predictions.jsonl` 的**离线重算分析**，无新推理（原生YOLO26n除外，见下）、无训练、未修改任何冻结证据。
> 动机：第三章"背景干扰/夜间"问题此前只有定性叙述与固定阈值事件数；本表给出逐时段的正式尺寸AP/AR量化。

## 属性来源与核验

- 官方 `bdd100k_labels.zip`（http://dl.yf.io/bdd100k/bdd100k_labels.zip ，2025-02-06版），SHA256 `7f1f9043c70a6ff0788a323cfb914aeced109a37b7150740d670a347c881394a`，189,638,612字节。
- val 逐图 scalabel JSON 持久化于 `data/BDD100K/label_source_2025/val_scalabel/`（10,000文件，gitignored）。
- 时段分布 night 3,929 / daytime 5,258 / dawn-dusk 778 / undefined 35，与项目冻结统计 `results/dataset_stats/bdd100k/stats.json` **逐数一致**（分布交叉核验通过）。

## 方法

`scripts/evaluate_timeofday_slices.py`：按官方属性把每份冻结预测按图像分入时段切片，复用 `evaluate_object_sizes.py` 的匹配与101点AP机制（同一代码路径，无独立实现）。输出 `batch1-r0-a1-k1-v10n.json`、`batch2-native-yolo11n-v8n-cb16-t0.json`。

注意：R0为官方P2结构的b32复训对照（ACR协议R0），非b64原版（其逐图预测已清理）；原生YOLO26n逐图预测本轮重生成，复现校验 `results/baseline/bdd100k-yolo26n-640-size-metrics-repro.json` 与冻结JSON差 ≤1.1e-5（cuDNN非确定性量级）。

## 首批数值（AP50-95）

| 模型 | day all | night all | Δ% | day small | night small | Δ% |
|---|---:|---:|---:|---:|---:|---:|
| 原生YOLO26n | 0.2614 | 0.2441 | -6.6 | 0.1018 | 0.0837 | -17.7 |
| YOLO11n | 0.2567 | 0.2431 | -5.3 | 0.0943 | 0.0709 | -24.8 |
| YOLOv8n | 0.2604 | 0.2443 | -6.2 | 0.0961 | 0.0798 | -16.9 |
| YOLOv10n | 0.2688 | 0.2485 | -7.5 | 0.1033 | 0.0848 | -17.9 |
| 官方P2@b32(R0) | 0.2768 | 0.2539 | -8.3 | 0.1209 | 0.0938 | -22.4 |
| L0(ACR+ACFR) | 0.2918 | 0.2650 | -9.2 | 0.1344 | 0.1114 | -17.1 |
| C-b16对照 | 0.2898 | 0.2686 | -7.3 | 0.1382 | 0.1126 | -18.5 |
| K1蒸馏 | 0.2944 | 0.2709 | -8.0 | 0.1355 | 0.1146 | -15.4 |
| T0教师 | 0.3237 | 0.2939 | -9.2 | 0.1538 | 0.1222 | -20.6 |

### 夜间相对增益（点估计）

- **L0 vs R0(P2)**：day small +11.21% → **night small +18.76%**（自适应互补融合在夜间收益放大，第四章机制证据）。
- **K1 vs L0**：day all +0.92% → **night all +2.24%**；day small +0.76% → night small +2.88%。
- **K1 vs C-b16（同口径对照）**：day small -2.00% → **night small +1.77%**（蒸馏small增益在夜间翻转出现，第五章机制证据）。
- K1 的夜间small相对退化 -15.4% 为全部九模型最小。

## 限制

- 切片指标为点估计，未做切片内配对Bootstrap（后续可用 `bootstrap_bdd_paired_ap.py` 加切片过滤）。
- dawn/dusk 与 undefined 切片数值在JSON中，未列入主表（778/35图）。
- 属描述性分析，不构成任何门或选型依据（未在任何切片上调参）。
