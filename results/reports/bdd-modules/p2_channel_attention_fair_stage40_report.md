# P2浅层通道注意力公平Stage 40报告

> 工作包：`P2-EARLY-CHANNEL-ATTENTION-FAIR-STAGE40`  
> 日期：2026-07-31（Asia/Shanghai）  
> 结论：同epoch整体容忍门、训练连续性和尺寸成熟度通过；允许精确恢复到epoch60。

## 1. 恢复与暂停完整性

- 从公平Stage 20的完整`last.pt`恢复，训练前确认904/904项迁移，并逐张量确认实时模型与checkpoint EMA的904个张量相等。
- epoch21训练loss约1.9/1.4，与epoch20连续，没有重新初始化。
- epoch40保存后按设计暂停；CSV共40行且全部有限，无OOM、NaN、Inf、early stopping或自动恢复。
- checkpoint中`epoch=39`、planned epochs=200，optimizer/scaler/EMA完整，可精确resume。
- epoch21–40耗时12,800.5秒（3.556小时）；累计有效训练时间7.090小时，官方同epoch累计7.003小时，约+1.24%。

## 2. 同epoch公平比较

| epoch40指标 | 官方P2 | 通道注意力 | 相对变化 |
|---|---:|---:|---:|
| Precision | 0.65610 | 0.66355 | +1.14% |
| Recall | 0.39626 | 0.39032 | -1.50% |
| mAP50 | 0.44070 | 0.43607 | -1.05% |
| mAP50-95 | 0.24584 | 0.24342 | -0.98% |

候选整体AP仍在-2%容忍范围内。候选mAP50-95从epoch20的0.23576升至epoch40的0.24342，增加0.00766；epoch30至40仍增加0.00349，最近10轮单轮增益全部为正，尚不能判定停滞。Recall仍低于官方同epoch，是Stage 60需要继续观察的主要风险。

## 3. 完整10k独立验证与尺寸成熟度

独立Ultralytics val：P约0.664、R约0.390、mAP50约0.436、mAP50-95约0.244，与训练CSV一致。

| 尺寸 | Stage 20 AP50-95 | Stage 40 AP50-95 | 相对变化 | Stage 40 AR50-95 |
|---|---:|---:|---:|---:|
| all | 0.238169 | 0.244886 | +2.82% | 0.428649 |
| small | 0.085917 | 0.088533 | +3.05% | 0.261950 |
| medium | 0.269401 | 0.278016 | +3.20% | 0.478388 |
| large | 0.473587 | 0.484901 | +2.39% | 0.626666 |

小目标AR也从0.254950升至0.261950（+2.75%）。官方P2正式run没有保存epoch40权重，因此本表只比较候选自身Stage 20→40成熟度，不能作为官方P2与候选的同epoch尺寸差分。

## 4. 证据

- `results.csv` SHA256：`8f710e16e08a44348ce19fc9f0cdaff90a4de4d537cba7bfee9573560b6407ae`。
- `best.pt`/`last.pt` SHA256：`4bf5443d149b926998a3522de18520f2db12ae7c31e24141ef8ba79f40b71392`。
- 尺寸JSON SHA256：`3dfe5308963f74b10c883757c395a08420e91118fbae3d585b2a6cfb9d16a887`。
- 逐图预测SHA256：`a0d21e8dd29289c02a2f1a3e0f775d59911bfb1225c9a061c74fde46d99819b4`。

## 5. 决策

Stage 40通过：整体指标相对官方同epoch退化小于2%，最近10轮仍持续改善，small/medium/large AP与AR均较Stage 20上升，额外训练成本约1.24%。下一步只从last精确恢复epoch41–60，并在epoch60保存后再次强制暂停；不授权越过epoch60。
