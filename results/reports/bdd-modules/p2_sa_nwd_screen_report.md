# Phase 2 SA-NWD 配对 50 epoch 首轮筛选报告

> 工作包：`P2-SA-NWD-SCREEN`  
> 日期：2026-07-29（Asia/Shanghai）  
> 结论：连续尺寸权重提高 small AP/AR，但 all 和 medium AP 退化，不进入 200 epoch；按冻结路线测试 small-only 降级版。

## 1. 配对公平性

- 两组均使用官方 YOLO26n-P2、原始 `yolo26n.pt`、固定 BDD100K 400/100、640、50 epoch、batch 32、seed 0 和物理 GPU0。
- 同一提交 `3b8eb2a1b54cd33b85ea6ceb8717c37995d76ff7`，除 `sa_nwd` 和 run name 外配置一致。
- control 使用原生 CIoU；candidate 使用 C=32、τ=1024、alpha 上限 0.8 的连续尺寸权重。
- 两组均完成 50/50 epoch，无 early stopping、OOM、NaN、Inf 或 fitness 自动恢复。

## 2. Ultralytics 指标

| 指标 | P2 control | P2+SA-NWD | 变化 |
|---|---:|---:|---:|
| 最佳 epoch | 40 | 40 | — |
| Precision | 0.53962 | 0.52228 | -0.01734 |
| Recall | 0.14626 | 0.15800 | +0.01174 |
| mAP50 | 0.14063 | 0.13762 | -0.00301 |
| mAP50-95 | 0.06737 | 0.06366 | -0.00371（-5.51%） |
| 有效训练时间 | 194.96 s | 200.15 s | +2.66% |

best 独立验证与 CSV 最佳点一致：control mAP50-95 约 0.0672，candidate 约 0.0637。

## 3. 尺寸专项评估

| 尺寸 | control AP50-95 | SA-NWD AP50-95 | 绝对变化 | 相对变化 |
|---|---:|---:|---:|---:|
| all | 0.067716 | 0.064143 | -0.003573 | -5.28% |
| small | 0.016634 | 0.017781 | +0.001147 | +6.90% |
| medium | 0.105803 | 0.098235 | -0.007568 | -7.15% |
| large | 0.190013 | 0.196215 | +0.006202 | +3.26% |

| 尺寸 | control AR50-95 | SA-NWD AR50-95 | 相对变化 |
|---|---:|---:|---:|
| all | 0.186840 | 0.183744 | -1.66% |
| small | 0.104344 | 0.106018 | +1.60% |
| medium | 0.271738 | 0.272810 | +0.39% |
| large | 0.299891 | 0.302145 | +0.75% |

SA-NWD 同时提高 small AP 和 AR，验证了定位损失方向；但 all AP 与 medium AP 明显下降，没有通过正式候选门槛。

## 4. 诊断与下一步

连续 `alpha=exp(-A/τ)` 在 COCO medium 下界 32×32 处仍为 0.3679，因此会明显改变边界附近的 medium 框。下一步按方案 11.3 冻结降级路线，把 NWD 严格限制为参考 640 尺度下 `A < 1024 px²`，中、大目标恢复纯 CIoU；small 区间内部仍保留连续 alpha 和 0.8 上限。

small-only 版本仍需相同 50 epoch、best 回验和尺寸评估。若不能保留 small 增益并消除整体退化，则降低 alpha 上限；仍无效才淘汰 SA-NWD。

## 5. Small-only 降级结果

small-only 完成 50/50 epoch，无自动恢复、NaN、Inf、OOM 或早停；最佳 epoch 39，Ultralytics mAP50-95 0.07006。相同尺寸专项协议结果如下：

| 尺寸 | control AP50-95 | small-only AP50-95 | 相对变化 |
|---|---:|---:|---:|
| all | 0.067716 | 0.063147 | -6.75% |
| small | 0.016634 | 0.017204 | +3.43% |
| medium | 0.105803 | 0.096376 | -8.91% |
| large | 0.190013 | 0.175913 | -7.42% |

small AR 相对 control 下降 2.02%。硬 small 阈值保留少量 small AP 增益，但没有消除整体退化，且损害 small AR，因此失败。

最后执行一个单变量快速衰减诊断：恢复连续权重，把 τ 从 1024 降到 512。这样 32×32 框的 alpha 从 0.3679 降到 0.1353，16×16 仍为 0.6065；若仍不能同时改善 small 和保持整体，则淘汰 SA-NWD，不继续参数搜索。

## 6. τ=512 最终诊断与路线判定

τ=512 完成 50/50 epoch，无自动恢复、NaN、Inf、OOM 或早停；最佳 epoch 40，Ultralytics mAP50-95 0.06273。

| 尺寸 | control AP50-95 | τ=512 AP50-95 | 相对变化 |
|---|---:|---:|---:|
| all | 0.067716 | 0.063298 | -6.52% |
| small | 0.016634 | 0.017028 | +2.37% |
| medium | 0.105803 | 0.097561 | -7.79% |
| large | 0.190013 | 0.190985 | +0.51% |

small AR 相对 control 下降 0.36%，all AR 下降 3.58%。快速衰减缩小了 small AP 增益，但仍未消除整体与 medium AP 退化。

最终三组 SA-NWD 诊断如下：

| 变体 | all AP | small AP | small AR | 判定 |
|---|---:|---:|---:|---|
| 连续 τ=1024 | -5.28% | +6.90% | +1.60% | 整体失败 |
| small-only | -6.75% | +3.43% | -2.02% | 失败 |
| 连续 τ=512 | -6.52% | +2.37% | -0.36% | 失败 |

SA-NWD 对 small AP 的方向性有效，但三种受控变体均无法同时满足 small AP/AR 改善和整体无明显退化。因此路线正式淘汰：不启动 200 epoch、不继续搜索 C/τ/alpha、不叠加注意力或其他网络模块。代码默认关闭，仅保留以复查负结果。活动模型恢复官方 P2，下一工作包为 640/960 分辨率实验。

## 7. 证据哈希

| 文件 | SHA256 |
|---|---|
| control `results.csv` | `b3f4464773aed9e18de39f6e153047fffbb1c720a5d8a9ec07f12bdade2457d0` |
| control `best.pt` | `65ecfb395242cc1c7c99bb42804373d8f56abd35ca12b839e1970ca7d442eab8` |
| candidate `results.csv` | `0efccf87b2722dd9ebf5bd715535ce20e7292f7793da5e7a10dbffd4c6273cfb` |
| candidate `best.pt` | `ded89765e2758897667ef1f13593bf3ca9693fee52de90f264e5a5a478dc29c4` |
| control 尺寸 JSON | `6f1b0b9ce5815b63667fc0f2990ef6e0b53bfccd4c9394c75c51413b76a7f678` |
| candidate 尺寸 JSON | `07a66a4fbd6869b79f27d639ed4e8e86d1d45aa5c56e0345395327e5b19fb6da` |
| small-only `results.csv` | `32ad119fc002dd5943a7b7dd323544201e11db4cbfe20ae3c64c84012a5a33f5` |
| small-only `best.pt` | `d2b7646a4bca5932ceff167c2bc37e4cdfb00836c35c1a88ef36365cd7f054fb` |
| small-only 尺寸 JSON | `e9d5a80476d18fba4332db455669e9860b499695e4ccf6bb8bd8eead68324a24` |
| τ=512 `results.csv` | `7f91a51b61d7a7f2ddb662450c6892cf6f0cd2e69a4b6deab9ea7117e9c8cd24` |
| τ=512 `best.pt` | `59d15f55052939557cdfb1359b4bfb8bd4ec726713d09038bc3ff39e9a0eb848` |
| τ=512 尺寸 JSON | `7acf9e71a26fa62189dd3cc227eff24d70f7e2e955ad98829b6e26b11e3a87d3` |
