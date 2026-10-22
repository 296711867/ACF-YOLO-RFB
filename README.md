# ACF-YOLO-RFB：基于改进 YOLO26 的道路小目标检测算法与 Jetson 边缘部署

> **English brief** — Road small-object detection on BDD100K, built on YOLO26n. The final model **ACF-YOLO-RFB** (2.29 M params) reaches **28.38 AP / 11.97 AP_small** (+11.12% / +34.80% relative over the YOLO26n baseline). A confidence-weighted output distillation (**ACF-YOLO-KD**, 28.70 AP, zero inference overhead) is studied for edge deployment. The final model is deployed on **Jetson AGX Orin** with TensorRT FP16 (5.12 ms pure inference) and a browser detection workbench sustaining **32 FPS** over a 30-minute run. Zero-shot transfer is evaluated on VisDrone / KITTI / RTTS. See §1 for results and [`docs/paper/THESIS_RESULTS.md`](docs/paper/THESIS_RESULTS.md) for the number-to-evidence map.

> **口径（2026-10-22，对齐论文 v2）**：最终模型 = **ACF-YOLO-RFB**（P2 分支 + 非对称容量重分配 ACR + 互补门控融合 CGF + 感受野增强 RFB）。知识蒸馏（ACF-YOLO-KD）是论文第五章的正面方法，因与 RFB 组合后未保持小目标增益而未选入最终部署；本仓库早期快照中"ACR 即最终方法、蒸馏/RFB 属负结果边界"的旧口径已被取代，对照见 [`docs/paper/THESIS_RESULTS.md`](docs/paper/THESIS_RESULTS.md) §5。

## 1. 核心结果

**表 1　BDD100K 10k val 逐步消融（640 输入，COCO 口径）**（论文表 4-2）

| 模型 | 参数量/10⁴ | GFLOPs | AP/% | AP50/% | **APs/%** | APs50/% | ARs/% |
|---|---:|---:|---:|---:|---:|---:|---:|
| YOLO26n（基准） | 237.68 | 5.32 | 25.54 | 46.41 | 8.88 | 24.09 | 23.71 |
| + P2 | 240.28 | 6.57 | 26.93 | 47.76 | 10.33 | 25.67 | 28.79 |
| + P2 + ACR | 226.99 | 8.10 | 28.05 | 49.67 | 11.08 | 27.89 | 30.80 |
| + P2 + ACR + CGF（ACF-YOLO） | 227.80 | 8.50 | 28.23 | 50.18 | 11.66 | 29.10 | 31.29 |
| **+ P2 + ACR + CGF + RFB（ACF-YOLO-RFB）** | **229.14** | 9.2 | **28.38** | **50.55** | **11.97** | **29.72** | 30.96 |

- **演进链**：P2 分支解决"小目标无高分辨率检测层"（APs 8.88→10.33）；ACR 把 P5 分支冗余通道移给 P2 分支，参数较 YOLO26n-P2 **减 5.53%**、APs 再 **+7.28%**；CGF 仅增 **8 132 参数**，APs 再 **+5.22%**；RFB 仅增 13 376 参数，APs 再 **+2.66%**。
- **相对基准**：ACF-YOLO-RFB 的 AP **+11.12%**、APs **+34.80%**（8.88→11.97），参数量反而少 3.59%。
- **稳定性**：三个随机种子下 AP/APs 均值 **28.57 / 12.15**；CGF 与 RFB 的小目标增益经配对自助法（1 000 次同图像配对 Bootstrap）检验支持。
- **蒸馏（第五章）**：ACF-YOLO-KD 整体 AP **28.70**（较无蒸馏相对 +1.66%），推理结构与代价完全不变；但 KD+RFB 组合实验整体 AP 点估计更高却未保持 RFB 的小目标增益，故最终部署模型定为**不含蒸馏的 ACF-YOLO-RFB**。
- **零样本迁移**：VisDrone / KITTI / RTTS 三集七模型全表见 [`results/visdrone-zeroshot/`](results/visdrone-zeroshot/)、[`results/kitti-zeroshot/`](results/kitti-zeroshot/)、[`results/rtts-zeroshot/`](results/rtts-zeroshot/)；改进模型具有跨场景迁移能力，但模块间相对排序随数据域改变（论文 4.4.10 如实呈现）。

数字与证据文件的一一对照（每个数字出自哪个 JSON）见 [`docs/paper/THESIS_RESULTS.md`](docs/paper/THESIS_RESULTS.md)。

## 2. 方法与代码位置

```
YOLO26n ──+ P2 检测分支 ──+ ACR 非对称容量重分配 ──+ CGF 互补门控融合 ──+ RFB 感受野增强
                                                            (= ACF-YOLO)          (= ACF-YOLO-RFB)
```

| 模块 | 一句话 | 配置 | 代码 | 测试 |
|---|---|---|---|---|
| P2 分支 | 4 倍下采样高分辨率检测层 | `configs/models/`（P2 系列） | — | `tests/` |
| ACR | P5 检测头 256→192、P2 检测分支 32→64 通道，纯 YAML | [`configs/models/yolo26n-p2-acr.yaml`](configs/models/yolo26n-p2-acr.yaml) | — | `tests/test_acr_model.py` |
| CGF | 部分残差细化单元 + 分组互补门控单元（+8 132 参数） | [`configs/models/yolo26n-p2-acr-acfr.yaml`](configs/models/yolo26n-p2-acr-acfr.yaml) | [`sdf/modules/acfr.py`](sdf/modules/acfr.py) | `tests/test_acfr*.py` |
| CGF 消融 | 去门控 / 去细化 | `…-nogate.yaml` / `…-norefine.yaml` | 同上 | 同上 |
| RFB | 三分支空洞卷积 + 残差（+13 376 参数） | [`configs/models/yolo26n-p2-acr-acfr-rfb.yaml`](configs/models/yolo26n-p2-acr-acfr-rfb.yaml) | [`sdf/modules/rfb.py`](sdf/modules/rfb.py)（YAML 注册名 `RFBlock`） | — |
| 输出蒸馏 | 学生一对多分支上按教师置信度加权（第五章） | 教师 [`configs/models/yolo26n-p2-s-teacher.yaml`](configs/models/yolo26n-p2-s-teacher.yaml) | `sdf/`（蒸馏损失） | `tests/` |
| 轻量化筛查 | 深度卷积/Ghost/混合五种方案对比（第五章） | `…-dw-*` / `…-ghost-*` / `…-h1~h3-hybrid.yaml` | — | — |

`upstream/` 为冻结的 Ultralytics 8.4.104 源码快照（AGPL-3.0），全部实验在 `PYTHONPATH=upstream:.` 下运行，保证版本可复现。

## 3. Jetson 部署与浏览器工作台（第六章）

平台 Jetson AGX Orin（TensorRT FP16）。三候选模型部署实测（论文表 6-2）：

| 指标 | ACF-YOLO | ACF-YOLO-RFB | ACF-YOLO-KD |
|---|---:|---:|---:|
| 纯推理延迟中位数/ms | 4.82 | 5.12 | 4.84 |
| 图像端到端吞吐/FPS | 61.34 | 60.07 | 61.10 |
| 视频端到端吞吐/FPS | 34.37 | 33.87 | 34.29 |

- Jetson 端与服务器端同一精度子集上整体/小目标 AP 相对差异 **< 0.25%**（论文表 6-4）。
- 浏览器工作台后端（含读帧、预览、完整留档）经输出编码与预览优化，五次吞吐中位数 **22.07 → 32.46 FPS**；最终模型连续运行 1 800 s 处理 **57 651 帧、平均 32.03 FPS**，录像与检测日志帧数一致（论文表 6-7~6-9）。
- 系统代码：[`yolo-system/`](yolo-system/)（纯 Python 标准库后端 + 原生前端；图片/视频检测、三模型对比、任务留档与 CSV 导出）。验收证据：`yolo-system/ACCEPTANCE.json` 与 `yolo-system/docs/evidence/`。

## 4. 论文章节 ↔ 仓库映射

| 论文章节 | 内容 | 代码/配置 | 证据目录 |
|---|---|---|---|
| 第三章 | 数据分析与基准 | `configs/data/bdd100k.yaml` | `results/dataset_stats/`、`results/baselines/` |
| 第四章 | P2/ACR/CGF/RFB | §2 表 | `results/acr-stage20/`、`results/acfr-stage20/`、`results/p29-context-screen/` |
| 4.4.8 | 与其他模型对比 | — | `results/reports/bdd-modules/` |
| 4.4.10 | 零样本迁移 | — | `results/{visdrone,kitti,rtts}-zeroshot/` |
| 第五章 | 轻量化筛查 | `…-dw-*`/`…-ghost-*`/`…-h*-hybrid.yaml` | `results/lightweight-*/`、`results/p28-simam/`、`results/p29-context-screen/` |
| 第五章 | 输出蒸馏 | `yolo26n-p2-s-teacher.yaml` | `results/kd-teacher/`、`results/kd-arms/`、`results/p31-kd-rfb/` |
| 第六章 | Jetson 部署与网页系统 | `yolo-system/` | `results/jetson-deploy/`、`yolo-system/docs/evidence/` |
| 全文 | 图表数据源 | — | `results/paper_figures/`、`results/paper-visuals/` |

## 5. 快速开始

```bash
# 环境（Python ≥3.10；训练在单卡 RTX 3090 / A100 上完成）
conda activate yolo26-sdf
python -m pip install --no-build-isolation --no-deps -e .
PYTHONPATH=upstream:. python scripts/verify_setup.py
PYTHONPATH=upstream:. python -m unittest discover -s tests -v   # 含 ACFR/RFB 模块测试

# 用最终权重直接验证（权重已入库）
PYTHONPATH=upstream:. python - <<'PY'
from ultralytics import YOLO
m = YOLO("weights/ACF-YOLO-RFB.pt")
print(m.info())   # 2.29M 参数
PY

# 数据集准备：BDD100K 官方 10k 检测标注（train 70k / val 10k），路径写入 configs/data/bdd100k.yaml
# 正式训练只能经启动器（校验配置/数据/checkpoint，防止覆盖证据）：
PYTHONPATH=upstream:. python scripts/run_experiment.py --help
```

网页工作台：`cd yolo-system && python server.py`（仅需 Python 标准库 + TensorRT 环境；详见 [`yolo-system/README.md`](yolo-system/README.md)）。

## 6. 实验证据与入库口径

- **入库**：全部关键指标 JSON（`*-size-val-ep*.json`、`*-summary.json`、`bootstrap-*.json`、`inference_*.json`）、冻结协议（`docs/protocols/`）、审计与分析报告（`results/reports/`）、最终权重 `weights/ACF-YOLO-RFB.pt`。
- **不入库**：逐图预测 `*.predictions.jsonl.gz`（每份 85–95 MB）、三种子/对照/教师权重、TensorRT 引擎、数据集本体。实验数字均可由入库 JSON 直接复核；原始预测保存在内部审计库。
- 研究纪律：协议先冻结、门数值训练前定、正式结果只从冻结日志与机器证据生成；`docs/paper/RESULTS.md` 为 ACR 阶段冻结审计记录（§1.1–§1.10），论文 v2 最终口径以 [`docs/paper/THESIS_RESULTS.md`](docs/paper/THESIS_RESULTS.md) 为准。

## 7. 负结果与设计决策

- **结构轻量化路线被数据否决**：TensorRT 逐层剖析显示通常被替换的高层卷积只占推理延迟 **2.20%**，五种轻量化方案中四种有延迟测量均未加速（参数降幅最高 34.57%）→ 转向不改变推理结构的知识蒸馏（第五章动机）。
- **KD+RFB 不组合**：组合模型整体 AP 点估计更高，但未保持 RFB 的小目标增益（且存在训练批大小差异）→ 以小目标性能为核心，最终部署模型定为 ACF-YOLO-RFB。
- 其余筛查性负结果（SDD、DWF、SA-NWD、通道注意力、透视分带路由、裁块、Haar 下采样等）保留在 `docs/paper/RESULTS.md` §4 与对应 `results/` 目录，供复查。

## 8. 许可与引用

- 本仓库以 **AGPL-3.0** 发布（`LICENSE`）；`upstream/` 为 Ultralytics 8.4.104 快照，遵循其原始 AGPL-3.0 许可与版权；`yolo-system/vendor/` 内第三方文件见其各自声明。
- 论文图表与叙述文本版权归作者所有，转载需授权。引用条目待论文发表后补充。
