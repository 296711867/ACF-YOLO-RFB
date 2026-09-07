# P2浅层通道注意力首轮完整BDD Stage 20诊断报告

> 工作包：`P2-EARLY-CHANNEL-ATTENTION-STAGE20-DIAG`
> 日期：2026-07-30（Asia/Shanghai）
> 结论：工程与趋势健康，但学习率计划不公平且checkpoint不可续；降级为诊断，重新执行公平Stage 20。

## 1. 完成状态

- 完整BDD100K 70k train / 10k val、640、batch64、seed0、GPU0。
- 完成20/20 epoch，累计11,866.5秒（3.296小时），平均9.89分钟/epoch。
- 无OOM、NaN、Inf、early stopping或自动恢复；20行CSV全部有限。
- 正常结束，元数据`status=completed`，GPU0已释放。

## 2. 指标与趋势

| 指标 | epoch 20 |
|---|---:|
| Precision | 0.64168 |
| Recall | 0.39521 |
| mAP50 | 0.43073 |
| mAP50-95 | 0.23881 |

最佳点为epoch20。mAP50-95从epoch10的0.22162升至0.23881，增加0.01719；末5轮单轮增益依次约0.00124、0.00120、0.00070、0.00085、0.00048，仍上升但边际趋缓。

## 3. 公平性问题

官方P2正式训练的epoch20 mAP50-95为0.23685，候选高0.00196（0.83%），但不能视为公平提升：

- 候选配置`epochs=20`，epoch20主学习率已降至0.001785。
- 官方P2配置`epochs=200`，epoch20主学习率仍为0.0271785。
- 候选正常结束后best/last被剥离optimizer/scaler，checkpoint epoch为-1，不能精确resume到40。

因此，本run只证明完整数据工程可行、成本可控和曲线健康，不支持模型优于官方P2的结论。

## 4. 证据哈希

| 文件 | SHA256 |
|---|---|
| `results.csv` | `98e4a20a9fd82d00b82fae9ccbf6337faaa67185c39c6d77b8c8956aff3e7ecc` |
| `best.pt` | `3418d8d28fad0cd6a293c024509e1608e8292c269087ddee880d39bd854a7dc6` |
| `last.pt` | `2126afd3195927dee4e4613655ed7a42f159e011d556b66ab17186d2d87b44f1` |

## 5. 修正方案

重新从原始`yolo26n.pt`启动候选，设置与官方P2相同的200-epoch学习率计划；正式启动器在epoch20 checkpoint保存后抛出受控阶段边界，跳过最终剥离并验证last含optimizer/scaler/EMA。只有该公平run完成独立/尺寸评估后，才决定是否精确恢复到epoch40。

修正机制已通过真实集成测试：计划3 epoch的smoke在epoch1保存后暂停，checkpoint保留optimizer/scaler/EMA且planned epochs仍为3；随后从epoch2精确恢复，并在epoch2保存后再次暂停，CSV连续、resume lock清理、历史和哈希记录正常。

公平Stage 20已于2026-07-30 21:59通过正式preflight后启动，commit为`d25d9cd8816e10b4d000c1f24936a6da6083f1bd`，tmux为`yolo26-p2-attn-fair-stage20`。训练已进入epoch1；启动器将在epoch20 checkpoint保存后强制暂停。
