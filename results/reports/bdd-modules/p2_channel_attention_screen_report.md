# P2浅层近恒等通道注意力配对50 epoch筛选报告

> 工作包：`P2-EARLY-CHANNEL-ATTENTION-SCREEN`
> 日期：2026-07-30（Asia/Shanghai）
> 结论：候选通过全部尺寸保留门，允许进入完整BDD 20 epoch阶段门；禁止直接200 epoch。

## 1. 配对公平性

- control为既有官方YOLO26n-P2固定子集50 epoch证据，candidate仅在P2浅层分支concat前增加近恒等通道门控。
- 两组配置除model/name外完全一致：同一原始`yolo26n.pt`、BDD100K 400/100、640、50 epoch、batch32、nbs64、seed0、AMP、优化器和增强。
- 两组都从原始权重迁移360项；候选仅增加4,160参数。
- control已有完整CSV、best和相同协议尺寸JSON，因此复用，不重复训练。

## 2. 训练与独立验证

两组均完成50/50 epoch，最佳epoch均为40；无early stopping、自动恢复、OOM、NaN或Inf。

| 指标 | 官方P2 control | 通道注意力 | 变化 |
|---|---:|---:|---:|
| Precision | 0.53962 | 0.43056 | -0.10906 |
| Recall | 0.14626 | 0.17166 | +0.02540 |
| mAP50 | 0.14063 | 0.15205 | +8.12% |
| mAP50-95 | 0.06737 | 0.07459 | +10.72% |
| 50 epoch时间 | 195.084 s | 195.739 s | +0.34% |

候选best独立val为P 0.434、R 0.171、mAP50 0.152、mAP50-95 0.0742，与训练best一致。Precision下降但Recall、mAP50和mAP50-95均提升；最终裁决以AP/AR尺寸门为准。

## 3. 尺寸专项评估

协议：100张val、COCO风格101点AP、IoU 0.50:0.95、conf0.001、iou0.7、maxDet300。

| 尺寸 | control AP50-95 | candidate AP50-95 | 相对变化 | control AR50-95 | candidate AR50-95 | 相对变化 |
|---|---:|---:|---:|---:|---:|---:|
| all | 0.067716 | 0.072422 | +6.95% | 0.186840 | 0.195749 | +4.77% |
| small | 0.016634 | 0.018237 | +9.64% | 0.104344 | 0.106015 | +1.60% |
| medium | 0.105803 | 0.116253 | +9.88% | 0.271738 | 0.275300 | +1.31% |
| large | 0.190013 | 0.212192 | +11.67% | 0.299891 | 0.356667 | +18.93% |

small AP超过+5%、small AR为正，且all/medium/large AP均提升，因此全部冻结门通过。

## 4. 门控与成本

- attention weight绝对值均值0.01379、最大值0.04041，确认门控参与训练。
- bias-only通道增益约0.9832–1.0131，仍保持温和近恒等调制。
- 参数仅+0.16%，50 epoch实测时间仅+0.34%，没有不可解释计算成本。

## 5. 证据哈希

| 文件 | SHA256 |
|---|---|
| candidate `best.pt` | `6b3ce1d88a14eec4d4dfded764a69c5b6a751371a7a2268e0cb01fa1d305b05e` |
| candidate `results.csv` | `04ce0ad9e82d560afa72f5a42d1999090509ab4d77e2fa8ddf5f246513c45eb0` |
| candidate尺寸JSON | `0f8eabfdb7ebc480b1e35cf04dae8bb50f4b3a29546d0d9239c6ce20fea7cc11` |
| candidate逐图预测 | `effbff3db6e1692ee92145e1939e46b5ab7826d0e3d393c26212145383b64664` |
| candidate训练配置 | `4a6c1a2a1b9dc9fab71a196337a21f5e5444ced45ea4a2004a70f94d971dd9b1` |
| control `best.pt` | `9c90688f7e93c9c599b998cb4ce388620aa77eae04df11df49dba5fb32cd10fe` |
| control尺寸JSON | `cf848b3e2e7ceb37b273f7368b116e8bce4cae0f88bf867911cdc3b7973d4b37` |

## 6. 决策

保留浅层通道注意力候选。下一步先按官方P2完整数据协议做batch64显存预检，再只启动20 epoch阶段门；按历史速度预计约3.5小时。20 epoch结束后必须先评估边际收益和尺寸AP/AR，不能自动续到40/60/200 epoch。
