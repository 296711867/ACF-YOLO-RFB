# P1 YOLO26n 640 基线中断记录

> 检查日期：2026-07-26（Asia/Shanghai）  
> run：`runs/detect/phase1/bdd100k-yolo26n-640-e200-seed0`  
> 训练冻结 commit：`87ba6eca2cd5f44c2ac93bfbab2d8435333459d8`  
> 结论：外部中断，不是早停；checkpoint 可以精确续训。

## 1. 时间与进度

- 启动时间：2026-07-25 16:58:02 CST。
- 最后权重时间：2026-07-25 18:25:26 CST。
- CSV 内累计时间：5,224.19 秒，即 1 小时 27 分 4 秒。
- 已完整结束：17/200 epoch；checkpoint 内部 `epoch=16` 为从 0 开始计数。
- 2026-07-26 07:15 检查时训练进程不存在，GPU0 已空闲。

## 2. 不是早停

配置 `patience=50`，但训练只完成 17 epoch，且第 17 epoch 的 mAP50-95 为当前最高值 0.22620。早停要求连续 50 epoch 无提升，因此本次停止不可能由 early stopping 触发。

最后 5 个 epoch：

| epoch | train box | train cls | train l1 | mAP50 | mAP50-95 | val box | val cls | val l1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 13 | 2.09891 | 1.28450 | 0.00360 | 0.40963 | 0.22199 | 1.91710 | 1.16692 | 0.00439 |
| 14 | 2.09935 | 1.28467 | 0.00360 | 0.41325 | 0.22377 | 1.91379 | 1.16334 | 0.00438 |
| 15 | 2.09274 | 1.28011 | 0.00359 | 0.41535 | 0.22518 | 1.91037 | 1.16173 | 0.00437 |
| 16 | 2.08973 | 1.27664 | 0.00359 | 0.41607 | 0.22572 | 1.90761 | 1.16069 | 0.00436 |
| 17 | 2.08646 | 1.27560 | 0.00359 | 0.41622 | 0.22620 | 1.90599 | 1.16015 | 0.00435 |

17 个 epoch 的 CSV loss/指标全部为有限值，无 NaN/Inf。

## 3. 中断诊断

- `experiment_metadata.json` 仍为 `status: starting`，没有进入启动器的 success 或 exception 收尾分支，说明进程被非正常外部终止。
- 系统自 2026-07-25 11:25 起未重启。
- 18:20–18:30 系统日志中未发现 OOM、Killed process、NVRM 或 Xid 记录。
- 训练时 GPU0 峰值显存约 15.4 GB，低于 24 GB，不支持显存溢出的判断。
- 精确外因没有被日志记录。结合前台命令约 90 分钟后消失，推测执行会话生命周期结束的可能性较高，但该项是推断，不作为已证实根因。

## 4. checkpoint 状态

- `last.pt` 包含 optimizer、scaler、EMA、更新次数和原始 `train_args`，目标 epoch 仍为 200。
- `best.pt` 与 `last.pt` SHA256 相同：`59caf446f87305db5d38eb81ccbabbcf9619734750bf870fbe30b1acdc01b064`。
- 当前最佳 epoch：17；mAP50 0.41622，mAP50-95 0.22620。
- 可以从下一 epoch（18）续训，无需重新开始。

## 5. 后续决策

已新增 `scripts/resume_experiment.py`：它校验 Git clean、原配置哈希、checkpoint optimizer/scaler/EMA、CSV epoch、GPU 和并发 lock，并记录 `resume_history.json`。续训放入独立 tmux 持久会话，从 epoch 18 使用原 run、last checkpoint、数据、batch、seed 和训练参数继续；完成后再执行独立 val 与尺寸专项评估。

## 6. 最终处理结果

受控续训于 2026-07-26 23:08 完成全部 200 epoch，未触发早停。独立 val 和尺寸专项评估均已通过，本中断已关闭；最终结论见 `results/reports/baselines/phase1_baseline_640_report.md`。
