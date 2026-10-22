# P25 基线扩充数据报告：YOLOv8n / YOLOv10n / YOLO11n 同协议对比

> 生成时间：2026-09-28（P25 收官后自动整理）
> 范围：第五章对比表新增的三条同协议外部基线（YOLOv8n、YOLOv10n，P25 新增；YOLO11n，P21 已有），并与本项目 L0-200 / K1-200 / T0 教师参照对照。
> 所有数字均出自冻结协议下的一次性评估 JSON（`results/baselines/*.json`），无人工转录；本报告不改动任何已冻结证据。

---

## 1. 摘要

- 三条外部基线 **all AP50-95 落在 0.2520–0.2625**，内部排序 **YOLOv10n > YOLOv8n > YOLO11n**，最大相对差距 +4.2%（v10n vs 11n）。
- 本项目学生底座 **L0-200（YOLO26n）all 0.2823**，对外部最强基线 v10n **领先 +7.5%**，对 11n 领先 +12.0%；**small 维度领先幅度放大到 +30.7% ~ +43.0%** —— 对比表锚点扩充目的达成：YOLO26n 的优势在同协议第三方基线面前成立，且优势主要来自小目标。
- v10n 的 E2E 免 NMS 头带来显著的召回优势（all AR50-95 **0.430** vs v8n 0.367 / 11n 0.363，相对 +17~18%），但转化为 AP 的增益有限（+2.7~4.2%），呈现"高召回、定位精度相对弱"的特性。
- 三基线在 medium / large 桶几乎持平（large 0.516–0.532），**相互差异主要由 small 桶驱动**（相对差 9.4%），与论文小目标主题直接呼应。

---

## 2. 评估协议（三模型逐字段一致）

| 项 | 值 |
|---|---|
| 数据 | BDD100K val，10,000 图，10 类，共 185,526 个 GT 实例 |
| 指标 | COCO-style 101-point AP，IoU 0.50:0.95，macro（类均值），maxDet 300 |
| 推理 | imgsz 640，batch 32，conf 0.001，iou 0.7（v10 E2E 头下架构性失效，仅为协议一致保留） |
| 尺寸桶 | small ≤ 32² = 1,024 px²；medium 1,024–9,216；large > 9,216（GT 分布：small 102,729 / medium 59,669 / large 23,128，占 55.4% / 32.2% / 12.5%） |
| 评估代码 | `scripts/evaluate_object_sizes.py`（与 P21–P23 全部证据同一份代码） |

尺寸桶 GT 分布本身即 BDD 的难点注脚：**55% 的真值实例是小目标**。

---

## 3. 主要结果

### 3.1 分尺寸 AP（size-val 协议，评估 `best.pt`）

| 模型 | all AP50-95 | all AP50 | small AP50-95 | small AP50 | medium AP50-95 | large AP50-95 |
|---|---|---|---|---|---|---|
| **YOLOv10n** | **0.2625** | **0.4717** | **0.0892** | **0.2384** | 0.2970 | **0.5316** |
| YOLOv8n | 0.2555 | 0.4613 | 0.0835 | 0.2281 | **0.2971** | 0.5240 |
| YOLO11n | 0.2520 | 0.4541 | 0.0815 | 0.2244 | 0.2917 | 0.5162 |

相对差异（all AP50-95）：v10n vs v8n **+2.7%**，v8n vs 11n **+1.4%**，v10n vs 11n **+4.2%**；
（small AP50-95）：v10n vs v8n +6.8%，v10n vs 11n **+9.4%**。

### 3.2 召回 AR

| 模型 | all AR50-95 | all AR50 | small AR50-95 | large AR50-95 |
|---|---|---|---|---|
| **YOLOv10n** | **0.4298** | **0.7222** | **0.2375** | **0.6474** |
| YOLOv8n | 0.3670 | 0.6336 | 0.1579 | 0.6303 |
| YOLO11n | 0.3630 | 0.6211 | 0.1524 | 0.6204 |

v10n 的 AR 优势远大于其 AP 优势（small AR50-95 相对 v8n **+50.5%**，而 small AP 仅 +6.8%）：E2E one-to-one 头保留更多候选，但其中相当一部分定位质量不足以折算进高 IoU 段 AP。论文引用 v10n 时建议同时给 AR，避免单看 AP 低估其检测召回风格。

### 3.3 逐类 AP50-95（val GT 分布 + 三模型对照）

| 类别 | GT 实例 | YOLOv8n | YOLOv10n | YOLO11n |
|---|---: |---:|---:|---:|
| car | 102,506 | 0.4545 | **0.4570** | 0.4554 |
| traffic sign | 34,908 | **0.3087** | 0.3076 | 0.3050 |
| traffic light | 26,885 | 0.2086 | **0.2103** | 0.2064 |
| person | 13,262 | **0.2642** | 0.2559 | 0.2587 |
| truck | 4,245 | 0.3921 | **0.4129** | 0.3864 |
| bus | 1,597 | 0.3961 | **0.4274** | 0.3891 |
| bike | 1,007 | 0.1882 | **0.1958** | 0.1829 |
| rider | 649 | 0.1746 | **0.1792** | 0.1661 |
| motor | 452 | 0.1671 | **0.1787** | 0.1682 |
| train | 15 | 0.0009 | 0.0000 | 0.0014 |

- **train 类仅 15 个实例，AP 无统计意义，任何解读都应排除该类**（其波动纯粹是采样噪声）。
- v10n 的优势集中在**车辆大类**（truck +5.3%、bus +7.9% 相对 v8n）与骑行者；person 反而略低于 v8n/11n。
- 三模型在绝对主导类 car（占 GT 55.3%）上几乎无差（0.4545–0.4570），进一步说明排序差异来自长尾类与小目标。

---

## 4. 训练过程

### 4.1 协议一致性（冻结配置）

三份训练配置（`configs/train/bdd100k-yolo{v8n,v10n,11n}-640-e200-b32-seed0.yaml`）逐字段一致，仅三处不同：初始权重、device、project/name。共同协议：

| 项 | 值 |
|---|---|
| imgsz / epochs / batch / nbs | 640 / 200 / 32 / 64 |
| optimizer / seed / deterministic / amp | auto / 0 / true / true |
| patience / workers | 50 / 8 |
| 数据 | `configs/data/bdd100k.yaml`（同一划分） |
| 增强 | ultralytics 默认套件（mosaic 1.0、close_mosaic 10、fliplr 0.5 等），无任何模型特定调参 |

### 4.2 初始权重指纹（双源下载、字节级交叉校验）

| 权重 | 来源 | SHA256（前 8 位） | 体积 |
|---|---|---|---|
| yolov8n.pt | ultralytics/assets 官方 v8.3.0 | `f59b3d83` | 6,549,796 B（COCO nc=80，3,157,200 参数） |
| yolov10n.pt | THU-MIG/yolov10 官方 v1.1 | `61b91ffc` | 11,448,431 B |
| yolo11n.pt | ultralytics 官方 | `0ebbc80d` | —（P21 冻结记录） |

### 4.3 训练时长与中断恢复

| 模型 | 起止（wall-clock） | 累计 GPU 时长* | 中断 | 训练内联 best mAP50-95 |
|---|---|---|---|---|
| YOLOv8n | 09-27 10:00 → 09-28 05:42 | 13.9 h | **1 次**：09-27 14:51 用户让卡，ep69 后干净暂停；20:35 恢复（EMA 355 张量逐一比对一致，无损续训） | 0.2643 @ ep188 |
| YOLOv10n | 09-27 10:06 → 09-28 04:48 | 18.7 h | 0 次（曾等待 v8n 首 epoch 缓存门控后启动） | 0.2620 @ ep198 |
| YOLO11n | 09-17 21:09 → 09-18 23:47 | 14.8 h | 1 次 resume（P21 期间） | 0.2594 @ ep190 |

\* 累计 GPU 时长按 results.csv time 列分段求和（resume 后 time 列重置，不能只读末行）。v8n 与 v10n 双卡并行时共享 CPU 数据加载管线，单 epoch 耗时不可横向比较。三者 best epoch 均落在 188–198，未触发 EarlyStopping，收敛充分。

注：训练内联 val 指标（如 v8n 0.2643）与 size-val 协议终评（0.2555）口径不同（后者为独立冻结评估代码），**引用一律以 size-val JSON 为准**。

---

## 5. 模型复杂度（BDD nc=10 微调后 checkpoint 实测）

| 模型 | 层数 | 参数量 | GFLOPs@640 | 检测头 |
|---|---:|---:|---:|---|
| YOLOv8n | 130 | 3,012,798 | 8.2 | 解耦头 + NMS |
| YOLOv10n | 224 | 2,710,940 | 8.4 | v10Detect E2E（免 NMS） |
| YOLO11n | 182 | 2,591,790 | 6.5 | 解耦头 + NMS |
| L0 / K1（YOLO26n，参照） | 340 | 2,417,428 | 10.3 | — |

三者体量同档（nano 级，2.6–3.0M 参数），v10n 以最少参数拿到最高 AP；本项目 L0 参数量最小、精度最高，但计算量最大（10.3 GFLOPs）。

---

## 6. 与本项目模型对照（同评估协议）

| 模型 | all AP50-95 | small AP50-95 | vs 最强外部基线 v10n（all / small） | 证据 |
|---|---:|---:|---|---|
| T0 教师 | 0.3137 | 0.1314 | +19.5% / +47.4% | `120951e` |
| K1-200（输出蒸馏） | 0.2870 | 0.1167 | +9.4% / +30.8% | `810fc83` |
| **L0-200（YOLO26n 底座）** | **0.2823** | **0.1166** | **+7.5% / +30.7%** | `3a270de` observed |
| YOLOv10n | 0.2625 | 0.0892 | — | `045e2f5` |
| YOLOv8n | 0.2555 | 0.0835 | −2.7% / −6.4% | `10fa64d` |
| YOLO11n | 0.2520 | 0.0815 | −4.0% / −8.6% | `5dc2bdb` |

L0-200 数字取自 `results/kd-arms/bootstrap-k1-vs-l0-ep200.json` 的 observed 段（与 size-val 同一份预测、同一 macro-AP 代码）。**结论**：在补齐三条同协议外部锚点后，YOLO26n 底座对当代 nano 级基线的领先（all +7.5~12.0%，small +30.7~43.0%）构成第五章对比表的完整外部参照系；教师—蒸馏链（T0→K1/L0）的增益叠加在其上。

---

## 7. 关键发现（论文可直接引用的表述要点）

1. **排序与量级**：同协议下 v10n (0.2625) > v8n (0.2555) > 11n (0.2520)，三者 all AP50-95 极差仅 1.05pp；说明 nano 档模型在 BDD 上的"代际差异"远小于架构对小目标的处理能力差异。
2. **差异由 small 桶驱动**：medium/large 桶三模型最大相对差 <3%，small 桶达 9.4%——外部基线的排序本质上就是小目标能力的排序。
3. **v10n 高召回特性**：E2E 头使 all AR50-95 提升约 17–18%（相对），AP 转化率低；对比表中建议并列 AP 与 AR 两列，防审稿人质疑"v10n 是否被低估"。
4. **train 类（15 实例）为纯噪声**，任何逐类结论应排除；逐类差异集中在车辆类（truck/bus，v10n 领先 5–8% 相对）。
5. **协议可信度**：三基线与全部本项目历史证据共用同一份评估代码、同一划分、同一推理设置；初始权重经双代理源字节校验；v8n 中断恢复经 EMA 逐张量校验（355/355 一致）——对比表数字可直接经受盲审质询。

---

## 8. 证据与可复现性清单

| 证据 | 路径 / 提交 |
|---|---|
| v8n 终评 JSON | `results/baselines/yolov8n-640-size-val-ep200.json`（`10fa64d`，链尾自动提交） |
| v10n 终评 JSON | `results/baselines/yolov10n-640-size-val-ep200.json`（`045e2f5`） |
| 11n 终评 JSON | `results/baselines/yolo11n-640-size-val-ep200.json`（`5dc2bdb`，P21） |
| 训练 run 目录 | `runs/detect/phase25-baseline-expansion/bdd100k-yolo{v8n,v10n}-640-e200-b32-seed0/`、`runs/detect/phase21-yolo11n-baseline/…/` |
| v8n 暂停/恢复记录 | `.project.git` `c45d941`（暂停）；run 目录 `resume_history.json`（恢复，checkpoint SHA `ebd8c0d9…`） |
| 冻结配置 | `configs/train/bdd100k-yolo{v8n,v10n,11n}-640-e200-b32-seed0.yaml`（权重 SHA 在文件头注释） |
| L0/K1/T0 参照 | `results/kd-arms/bootstrap-k1-vs-{l0,c-b16}-ep200.json`、`PROJECT_STATUS.md` §1 P23 段 |
| 链日志 | `/tmp/p25_v8n_chain.log`、`/tmp/p25_v10n_chain.log`（临时，不入库） |

复现评估（任一基线）：

```bash
PYTHONPATH=upstream:. python scripts/evaluate_object_sizes.py \
  --model runs/detect/phase25-baseline-expansion/bdd100k-yolov8n-640-e200-b32-seed0/weights/best.pt \
  --data configs/data/bdd100k.yaml \
  --output /tmp/recheck-yolov8n.json --device 0
```

（正式运行只用 `scripts/run_experiment.py` / `scripts/resume_experiment.py`，见项目纪律。）
