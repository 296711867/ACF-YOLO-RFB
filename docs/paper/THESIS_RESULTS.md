# THESIS_RESULTS：论文 v2 数字 ↔ 仓库证据对照

> 本文件是公开仓库的**数字权威入口**（2026-10-22，对齐论文《基于改进 YOLO26 的道路小目标检测算法与系统研究》v2 完整版）。
> 每个论文数字在本仓库的证据文件路径。`docs/paper/RESULTS.md` §1.1–§1.10 为 ACR 阶段冻结审计记录，其"最终方法"表述已被本文件口径取代。

## 1. 主表（论文表 4-2，逐步消融，BDD100K val 10k / 640 / COCO 口径）

| 模型 | AP/% | APs/% | 参数量/10⁴ | 证据文件 |
|---|---:|---:|---:|---|
| YOLO26n（基准，b64） | 25.54 | 8.88 | 237.68 | `results/baselines/`（基准 val 记录） |
| + P2 | 26.93 | 10.33 | 240.28 | `results/p2/` |
| + P2 + ACR（b32） | 28.05 | 11.08 | 226.99 | `results/acr-stage20/full200_summary.json`、`r0-size-val-ep200.json` |
| + P2 + ACR + CGF（ACF-YOLO） | 28.23 | 11.66 | 227.80 | `results/acfr-stage20/a1-size-val-ep200.json`、`full200_summary.json` |
| + P2 + ACR + CGF + RFB（ACF-YOLO-RFB） | 28.38 | 11.97 | 229.14 | `results/p29-context-screen/rfb-size-val-ep200.json` |

- ACR 相对 YOLO26n-P2：参数 −5.53%、AP +4.15%、APs +7.28%、ARs +6.99% → `results/acr-stage20/bootstrap-acr-vs-official-ep200.json`
- CGF 组件消融（去门控 ng / 去细化 nr）：论文表 4-4 → `results/acfr-stage20/ng-size-val-ep200.json`、`nr-size-val-ep200.json`、`bootstrap-ng-vs-acr-ep200.json`、`bootstrap-nr-vs-acr-ep200.json`
- widen-only 对照（论文表 4-3）：`results/acr-stage20/bootstrap-acr-vs-widen-ep200.json`
- 三种子（论文表 4-6/4-7）：ACF-YOLO → `results/acfr-stage20/three_seed_summary.json`（含 l0-s1/l0-s2）；RFB → `results/p29-context-screen/rfb-s1-size-val-ep200.json`、`rfb-s2-size-val-ep200.json`、`bootstrap-rfb-s1-vs-l0-s1-ep200.json`、`bootstrap-rfb-s2-vs-l0-s2-ep200.json`、`bootstrap-rfb-3seed-pooled-ep200.json`
- 推理延迟（论文表 4-10，PyTorch FP32 与 TensorRT FP16）：`results/acr-stage20/inference_benchmark_gpu0_batch{1,16}.json`、`inference_trt_fp16_gpu0_batch{1,16}.json`；CGF/RFB TRT 对比 → `results/p29-context-screen/speed_trt_b16_l0_vs_rfb.json`（及 run2 交换顺序复测）

## 2. 蒸馏（第五章）

| 模型 | AP/% | 说明 | 证据 |
|---|---:|---|---|
| ACF-YOLO-KD（k1，置信度加权输出蒸馏） | 28.70 | 较无蒸馏 +1.66%（相对），推理代价不变 | `results/kd-arms/k1-size-val-ep200.json`、`bootstrap-k1-vs-l0-ep200.json`、`bootstrap-k1-vs-c-b16-ep200.json` |
| 蒸馏策略对比（k1 输出 / k2 特征 / k3 联合，论文表 5-3） | — | stage20 筛查 | `results/kd-arms/k{1,2,3}-size-val-ep20.json` |
| KD+RFB 组合（论文表 5-8，未选入最终部署） | — | 整体 AP 点估计更高、小目标增益未保持 | `results/p31-kd-rfb/k1-rfb-size-val-ep200.json`、`bootstrap-k1-rfb-vs-{k1,l0-rfb,c-b16}-ep200.json` |
| 教师模型（YOLO26s-P2） | — | — | `results/kd-teacher/`、`configs/models/yolo26n-p2-s-teacher.yaml` |
| 轻量化方案对比（论文表 5-1/5-2） | — | 五方案四种无加速 | `results/lightweight-unit-screen/`、`results/lightweight-runtime/`、`results/lightweight-h3/`、`results/lightweight-position-ablation/` |

## 3. 零样本迁移（论文表 4-11/4-12/4-13）

| 数据集 | 证据 |
|---|---|
| VisDrone val | `results/visdrone-zeroshot/` + 仓库根 `results/*-size-val-visdrone.json` |
| KITTI（标注图像） | `results/kitti-zeroshot/`（七模型，含 `k1-rfb-size-val-kitti.json`） |
| RTTS（雾天全量） | `results/rtts-zeroshot/` |

## 4. Jetson 部署与网页系统（第六章）

| 论文数字 | 证据 |
|---|---|
| 表 6-2 延迟/吞吐（4.82/5.12/4.84 ms 等） | `results/jetson-deploy/`（2026-10-17 AGX Orin 轮） |
| 表 6-4 Jetson vs 服务器精度一致性（<0.25%） | `results/jetson-deploy/` |
| 表 6-7/6-8 网页分阶段耗时与优化（22.07→32.46 FPS） | `yolo-system/docs/evidence/`、`yolo-system/experiments/` |
| 表 6-9 持续运行（1 800 s / 57 651 帧 / 32.03 FPS） | `yolo-system/docs/evidence/`、`yolo-system/ACCEPTANCE.json` |

## 5. 口径变更对照（旧 → v2）

| 旧口径（2026-10-22 前快照，含 `docs/paper/RESULTS.md` 与 `PROJECT_STATUS.md` 历史部分） | v2 口径（本文件） |
|---|---|
| "最终方法 = ACR" | ACR 为第四章第一步；最终模型 = **ACF-YOLO-RFB** |
| "蒸馏"列入负结果边界（RESULTS.md §4） | 蒸馏为第五章正面方法（ACF-YOLO-KD，AP 28.70），只是未选入最终部署 |
| RFB 未出现（阶段内未见） | RFB 为最终模型组成部分（+2.66% APs） |
| `sdf/` 全部标注"已关闭路线" | `sdf/modules/acfr.py`（CGF）与 `sdf/modules/rfb.py`（RFB）为最终模型在用模块；其余模块确为关闭路线 |

## 6. 不入库内容

逐图预测 `*.predictions.jsonl.gz`、三种子/对照/教师 `best.pt`、TensorRT 引擎（`.engine`）、数据集本体（BDD100K/VisDrone/KITTI/RTTS）、`runs/` 训练过程产物。上述内容保存于内部审计库；入库 JSON 足以复核论文全部数字。
