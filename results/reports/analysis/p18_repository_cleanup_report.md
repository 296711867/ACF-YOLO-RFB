# P18期仓库清理报告（2026-08-30）

> 执行会话：2026-08-30，ACR 200轮全训进行中（GPU1）。延续P16清理纪律：可重建数据删除并记录再生路径；唯一证据保留。

## 删除项

| 项 | 大小 | 性质 | 再生路径 |
|---|---|---|---|
| Python/pytest/ruff缓存目录9个 | ~6MB | 可重建 | 自动生成 |
| `results/haar-downsample-stage20/r1-size-val.predictions.jsonl` | 273MB | 已关闭路线（P18A）的原始逐图预测；指标级证据`r1-size-val.json`与`summary.json`已规范保留 | checkpoint在`runs/detect/phase18-hwd-stage20/`，用`evaluate_object_sizes.py`重推理约2小时 |
| `data/BDD100K/label_source/`（lance标注源） | 4.4GB | 官方GitHub可重新获取的数据快照；YOLO格式标签360MB已转换在`data/BDD100K/labels/`，训练评估不受影响 | 官方bdd100k仓库重新clone，`scripts/prepare_bdd100k.py`引用该路径 |

合计释放约4.7GB，项目由28GB降至24GB。

## 明确保留项

- ACR全部证据：ep20/ep40预测JSONL与Bootstrap、stage20/stage40机器证据、协议、报告（当前正向路线，200轮全训进行中）。
- 全部checkpoint与唯一训练日志：含已关闭路线（SDD/DWF/SA-NWD/蒸馏/RDCI/PASCP/HWD）与P18A/P18B各run——负结果复查依赖。
- 规范副本：`results/perspective-router-final/`（1.65GB）等。
- `.worktrees/`：按`docs/REPOSITORY_MAP.md`第1节条件继续保留（P2/P4/P5/P6唯一日志、P7审计根、P13/P14逐图预测）。
- `upstream/`冻结源码、全部配置、协议、报告与状态文档。

## 状态文档同步

- `SESSION_HANDOFF.md`自P16后未更新，已刷新至P18B中期状态（ACR过门、ep40确认、200轮全训中、本清理记录）。
- `PROJECT_STATUS.md`结论区已含P18A关闭与P18B过门/确认段落（此前提交已更新）。

## 2026-09-06追加清理

- 删除ACR中间checkpoint预测3份（ep20的`r1-size-val.predictions.jsonl`、ep40的`r0/r1-size-val-ep40.predictions.jsonl`，共约800MB）：对应Bootstrap结果JSON全部保留；checkpoint在`runs/`与`.worktrees/`，可由`evaluate_object_sizes.py`重推理再生（每份约15分钟）。
- 保留全部200轮终评预测（R0/ACR/widen/seed1，及完成后seed2）作为规范副本。
- 删除34个Python缓存目录。
