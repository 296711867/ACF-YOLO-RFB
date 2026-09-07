# VisDrone YOLO26n/P2 Stage 20 验收报告

> 日期：2026-08-04（Asia/Shanghai）  
> 工作包：`VISDRONE-PAIRED-STAGE20`  
> 结论：两组训练和可续训检查点均健康；官方 P2 未通过冻结保留门，不续训至 epoch 100。

## 1. 公平设置与训练证据

两组均使用 VisDrone2019-DET train/val、同源 COCO `yolo26n.pt` 初始化、640、batch16/nbs64、seed0 和 100 epoch 学习率计划，并在完成 epoch 20 checkpoint 保存后暂停。只使用物理 GPU0，未触碰 GPU1 外部任务。

| 项目 | 原生 YOLO26n | 官方 YOLO26n-P2 |
|---|---:|---:|
| 检测层 | P3/P4/P5 | P2/P3/P4/P5 |
| Stage 20 累计时间 | 797.039 s | 1711.670 s（+114.75%） |
| 阶段最佳 mAP50-95 / epoch | 0.14195 / 15 | 0.13976 / 19 |
| `best.pt` SHA256 | `a93d54265aeab8ce1dde3176dd60eb84043acb11cc670e8af8862e3954520407` | `a2bc80d7f3f9738be09d9f120a23975ed86367a1f3994639d501c1f9d9fe3995` |
| `last.pt` SHA256 | `e179a1d7d628bfcf22cb74a477346a4226d670b957f235fdc5b290bd3ad1045d` | `feb35ab75e1f8df001dd0036f54d355bdd980221854c68ca87fbab5cb3bc0fb2` |

两个 `last.pt` 均为内部 epoch 19（已完成第 20 轮），保留 100 epoch 计划、EMA、optimizer、scaler 和训练参数；元数据均为 `paused_at_stage`。日志未发现 OOM、TaskAlignedAssigner CPU 回退、NaN、Inf、traceback 或自动恢复。batch16 预检显存为原生约 9.03 GB、P2 约 22.7 GB。

## 2. 独立验证

两组各自 `best.pt` 在同一 548 图 val 上以 imgsz640、batch16、conf0.001、IoU0.7、maxDet300 独立复验：

| 模型 | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| 原生 YOLO26n | 0.371 | 0.300 | 0.263 | 0.142 |
| 官方 YOLO26n-P2 | 0.378 | 0.293 | 0.256 | 0.140 |

## 3. COCO 尺寸口径

| 范围 | 原生 AP / AR | P2 AP / AR | AP 相对变化 | AR 相对变化 | 阶段门 |
|---|---:|---:|---:|---:|---|
| all | 0.144007 / 0.283725 | 0.141884 / 0.270156 | -1.47% | -4.78% | AP 通过 |
| small | 0.073429 / 0.193117 | 0.075710 / 0.179755 | +3.11% | -6.92% | AP、AR 均失败 |
| medium | 0.214769 / 0.399249 | 0.209470 / 0.381062 | -2.47% | -4.56% | AP 通过 |
| large | 0.363359 / 0.541744 | 0.313153 / 0.495136 | -13.82% | -8.60% | AP 失败 |

冻结门要求 small AP 至少相对 +5%、small AR 正提升、all AP 下降不超过 2%、medium/large AP 下降不超过 5%。P2 仅满足 all 和 medium AP 条件，因此不能恢复到 epoch 100。

原始结果：

- `results/visdrone-stage20/yolo26n-size-val.json`，SHA256 `1971bcf0ec6595dcce8bd7a454bab4804b869c83839879f375ab869dfa37e13f`。
- `results/visdrone-stage20/yolo26n-p2-size-val.json`，SHA256 `7ae6873b5b4508152f376b0619da3e026ae7fcbaf61001ce0caf869f9df256c1`。

## 4. VisDrone 官方兼容口径

使用官方原始 val annotations、ignored-region 逻辑和 maxDets 1/10/100/500；指标单位为百分数。

| 模型 | AP50-95 | AP50 | AP75 | AR100 | AR500 |
|---|---:|---:|---:|---:|---:|
| 原生 YOLO26n | 15.2394 | 29.5634 | 13.6797 | 21.8755 | 31.8573 |
| 官方 YOLO26n-P2 | 15.5235 | 29.6951 | 14.1269 | 21.3006 | 32.0645 |
| P2 相对变化 | +1.86% | +0.45% | +3.27% | -2.63% | +0.65% |

P2 在官方总 AP 上略有收益，主要来自 pedestrian 和 car，但 bicycle、tricycle、truck 等类别退化。该结果不足以推翻预先冻结的尺寸保留门。

官方格式预测树 SHA256：原生 `e12317a516d16c9db8611d89a0fc488c09769f6fd8cc1f5279363967e1a72369`，P2 `09b703fee63e84095d012fd2bac62d597deaf015fd4d02d15df56d79ae5062e0`。

## 5. 决策与下一步

1. 关闭 VisDrone P2 Stage20 路线，不恢复 P2 `last.pt` 到 epoch 100，也不在 test-dev 上评估失败候选。
2. 保留两个完整 Stage20 checkpoint 和所有评估证据，便于复查；P2 结果可作为“额外高分辨率头并非稳定提升所有小目标类别”的消融。
3. 下一训练候选应以原生 YOLO26n VisDrone control 为基座；先明确新的单变量方案和阶段门，再决定是否续训原生 control 或做配对实验。
4. test-dev 继续保持未使用，等最终模型选择冻结后只评估一次，避免对测试集迭代调参。
