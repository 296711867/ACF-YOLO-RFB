# P2浅层通道注意力公平Stage 20报告

> 工作包：`P2-EARLY-CHANNEL-ATTENTION-FAIR-STAGE20`
> 日期：2026-07-31（Asia/Shanghai）
> 结论：健康性、同epoch整体容忍门和上升趋势通过；允许精确恢复到epoch40。

## 1. 公平协议与暂停状态

- 候选与官方P2使用相同BDD100K 70k/10k、640、batch64、seed0、原始权重、增强和200-epoch学习率计划。
- 启动器在epoch20 checkpoint保存后强制暂停，没有进入epoch21或最终strip流程。
- 20行CSV全部有限，无OOM、NaN、Inf、early stopping或自动恢复。
- last中`epoch=19`、planned epochs=200，optimizer/scaler/EMA均存在，可精确resume。
- 本阶段耗时12,723.3秒（3.534小时），官方同阶段12,577.4秒，约+1.16%。

## 2. 同epoch公平比较

| epoch20指标 | 官方P2 | 通道注意力 | 变化 |
|---|---:|---:|---:|
| Precision | 0.64207 | 0.64937 | +0.00730 |
| Recall | 0.39013 | 0.38922 | -0.00091 |
| mAP50 | 0.42797 | 0.42662 | -0.32% |
| mAP50-95 | 0.23685 | 0.23576 | -0.46% |

候选整体AP在-2%容忍范围内。候选mAP50-95从epoch10的0.21711升至epoch20的0.23576，增加0.01865；相对官方差距从epoch10的-1.17%缩小到epoch20的-0.46%。末5轮增益均为正，尚未停滞。

## 3. 完整10k验证与尺寸绝对画像

独立Ultralytics val：P 0.649、R 0.390、mAP50 0.426、mAP50-95 0.236，与CSV一致。

| 尺寸 | AP50-95 | AP50 | AR50-95 | AR50 |
|---|---:|---:|---:|---:|
| all | 0.238169 | 0.430243 | 0.423560 | 0.704469 |
| small | 0.085917 | 0.219660 | 0.254950 | 0.540089 |
| medium | 0.269401 | 0.488928 | 0.472060 | 0.755566 |
| large | 0.473587 | 0.679895 | 0.621134 | 0.818371 |

官方P2正式run未保存epoch20权重，因此无法获得官方同epoch尺寸指标；上表只描述候选当前成熟度，不能与官方e200尺寸指标做阶段门差分。

## 4. 门控状态与证据

- attention weight绝对值均值0.10871、最大值1.11523；bias-only增益约0.9696–0.9910，门控已参与训练。
- `results.csv` SHA256：`51934dc3a828aae93b8924351ceb829e5eee4dc4ea2cc7adf839bdb161a11734`。
- `best.pt`/`last.pt` SHA256：`1467a02b775cdbfd718010e99aff6d0a4beb02302c0ae7803099162f5c4b93e5`。
- 尺寸JSON SHA256：`5b7b590b62432299381cd1fd081401e488cf2bab257229ae0dd6e1bf65d78735`。
- 逐图预测SHA256：`ffb8a547e2291b0fa062c40ddebd3e48909c00b0a27e2ebe294b1c1b127f9601`。

## 5. 决策

Stage 20通过：训练健康、整体退化小于2%、相对差距缩小且曲线继续上升。下一步从last精确恢复epoch21–40，并在epoch40保存后再次强制暂停；不授权越过epoch40。

首次Stage40 resume在epoch21完成前发现字符串`pretrained`覆盖checkpoint EMA，loss回到初始化水平且日志仅迁移360/904项；已立即Ctrl-C。Stage20 CSV仍20行，last SHA256仍为`1467a02b...`，证据未受污染。Trainer修复为在resume参数所有权处强制`pretrained=False`，并在`on_train_start`逐张量验证实时模型等于checkpoint EMA。真实集成确认904/904项迁移、904张量精确匹配、epoch2连续训练并再次暂停成功；正式Stage40于08:53重新启动，epoch21权重与loss连续。
