# VisDrone类别权重 Stage 20 验收报告

> 日期：2026-08-04（Asia/Shanghai）  
> 工作包：`VISDRONE-CLASS-WEIGHT-STAGE20`  
> 结论：工程与整体AP有收益，但small AP/AR核心门失败；不续训至epoch100。

## 1. 训练与健康性

候选相对原生YOLO26n control仅设置`cls_pw=0.25`，其余model、COCO初始化、data、640、batch16/nbs64、seed0、100-epoch调度、增强和优化器相同。正式训练在clean commit `24c75c8`启动并于epoch20保存后暂停。

| 项目 | 原生control | cls_pw=0.25 | 变化 |
|---|---:|---:|---:|
| Stage20时间 | 797.039 s | 789.533 s | -0.94% |
| 峰值显存 | 约9.03 GB | 约9.03 GB | 无增量 |
| 训练中最终mAP50-95 | 0.14163 | 0.14598 | +3.07% |
| 参数/GFLOPs/推理路径 | 2,376,786 / 5.2 / 原生 | 相同 | 无增量 |

候选`best.pt`与`last.pt`均为epoch20，SHA256为`0f190feb2522d73f010140676824222612244b88815e2747bc5f15f4b164f95e`。`last.pt`保留100-epoch计划、`cls_pw=0.25`、EMA、optimizer和scaler；日志无OOM、TaskAlignedAssigner CPU回退、NaN、Inf、traceback或自动恢复。

## 2. 独立验证

同一548图val、imgsz640、batch16、conf0.001、IoU0.7、maxDet300：

| 模型 | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| 原生control | 0.371 | 0.300 | 0.263 | 0.142 |
| cls_pw=0.25 | 0.397 | 0.302 | 0.269 | 0.146 |

## 3. 尺寸与稀有类别阶段门

| 范围 | control AP / AR | 候选 AP / AR | AP相对变化 | AR相对变化 | 结果 |
|---|---:|---:|---:|---:|---|
| all | 0.144007 / 0.283725 | 0.148289 / 0.277283 | +2.97% | -2.27% | all AP通过 |
| small | 0.073429 / 0.193117 | 0.075208 / 0.184068 | +2.42% | -4.69% | AP与AR失败 |
| medium | 0.214769 / 0.399249 | 0.224907 / 0.400634 | +4.72% | +0.35% | 通过 |
| large | 0.363359 / 0.541744 | 0.391637 / 0.605465 | +7.78% | +11.76% | 通过 |

目标类别all AP变化：bicycle +2.81%、tricycle -1.61%、awning-tricycle +1.55%，满足三类中两类提升，但提升幅度有限。冻结门要求small AP至少+3%且small AR为正提升，两项均失败，因此整体/中大目标收益不能使候选晋级。

尺寸原始结果：`results/visdrone-class-weight-stage20/yolo26n-clspw025-size-val.json`，SHA256 `222e7f9d7efc62e6e23149b76cd640782b5e0242063977f1c0be497c357b9a39`。

## 4. 官方兼容口径

| 指标（%） | 原生control | cls_pw=0.25 | 相对变化 |
|---|---:|---:|---:|
| AP50-95 | 15.2394 | 15.6789 | +2.88% |
| AP50 | 29.5634 | 30.4234 | +2.91% |
| AP75 | 13.6797 | 14.0561 | +2.75% |
| AR100 | 21.8755 | 21.6587 | -0.99% |
| AR500 | 31.8573 | 31.3567 | -1.57% |

官方口径同样表现为AP上升、召回下降，与内部small AR结论一致。预测树SHA256为`f47dd50302561789de8d804774d19ebe666574787cca9b460f67b40655c43aed`；官方结果SHA256为`6734ca14c8ec4580be8e9c27b1c38ad419dca62ac8cf0bad50101b5b8aa07524`。

## 5. 决策

1. 类别权重路线关闭，不搜索其他`cls_pw`，不恢复候选到epoch100，也不进入test-dev。
2. 负结果保留：温和逆频率权重可以无成本提升宏AP和中大目标，但当前实现降低small及官方召回，不满足小目标主目标。
3. VisDrone活动模型恢复原生YOLO26n control。下一步从其健康epoch20 `last.pt`受控恢复并连续训练至epoch100，再用最终best执行val双口径和一次性test-dev评估。
