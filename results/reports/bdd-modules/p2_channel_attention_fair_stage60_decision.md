# P2浅层通道注意力Stage 60与直跑决策

> 工作包：`P2-EARLY-CHANNEL-ATTENTION-FAIR-STAGE60`  
> 日期：2026-08-02（Asia/Shanghai）  
> 结论：Stage 60健康通过；按用户决定取消后续每20轮暂停，允许从epoch61直接续至计划终点200。

## 1. Stage 60完整性

- epoch41–60于2026-07-31 17:03按设计暂停，CSV共60行且全部有限。
- `last.pt`中`epoch=59`、planned epochs=200，optimizer/scaler/EMA完整，可精确恢复。
- 无OOM、NaN、Inf、early stopping或自动恢复；训练loss与前一阶段连续。
- Stage 60 checkpoint SHA256：`efe514ea323c0e309d4343f44a2b2a0c992c80248d969f1fd5808b4ef8aaa9bc`。
- Stage 60 CSV SHA256：`e2c73d32f257df326808514a457a95aa6479550d9d090a681d7affa0f5b5e4d8`。

## 2. 同epoch公平比较

| epoch60指标 | 官方P2 | 通道注意力 | 相对变化 |
|---|---:|---:|---:|
| Precision | 0.65908 | 0.66535 | +0.95% |
| Recall | 0.40275 | 0.40283 | +0.02% |
| mAP50 | 0.44829 | 0.44765 | -0.14% |
| mAP50-95 | 0.25100 | 0.25013 | -0.35% |

候选与官方同epoch差距从Stage 40的-0.98%缩小到-0.35%。候选mAP50-95从epoch40到60增加0.00671；最近10轮虽有两次小于0.0002的正常单轮波动，但净增0.00225，未出现持续停滞。累计有效训练时间10.602小时，官方同epoch10.511小时，成本约+0.86%。

## 3. 协议调整

原计划在60之后继续每20轮暂停。用户于2026-08-02明确要求不再采用20轮分段，并希望直接跑完。Stage 60曲线、整体公平指标和checkpoint完整性支持这一调整，因此后续恢复不再传入`--stop-epoch`：

- 从epoch61精确恢复，保持原始200-epoch调度、optimizer、scaler和EMA。
- 允许正常训练到epoch200；`patience=50`继续生效，若连续50轮fitness没有改善，可按原训练协议自动早停。
- 仍保留tmux持久化、resume lock/history、904张量精确恢复校验、最终独立验证和尺寸分组评估。
- 不改变模型、数据、batch、seed、增强或损失配置。

按近期速度估算剩余140轮约需24–25小时。训练结束或自动早停后，再统一做完整10k独立验证、small/medium/large评估及与官方P2-640最终结果对比。
