# Phase 2 DWF Variant A 配对 50 epoch 筛选报告

> 工作包：`P2-DWF-DIAG-A-SCREEN`  
> 日期：2026-07-29（Asia/Shanghai）  
> 结论：nearest+全局权重+局部门控提高 small AP，但整体退化明显，不进入 200 epoch；继续 Variant B 去除局部门控。

## 1. 公平性与完成状态

- 与既有 batch 32 P2 control 共用原始 `yolo26n.pt`、固定 BDD100K 400/100、640、50 epoch、seed 0 和 GPU0。
- 只把完整 DWF 的 DySample 替换为 nearest，保留全局权重、局部门控和轻量融合。
- 不包含 SDD 或 SA-NWD。
- 启动提交：`de287f2af565913c06bc486156971009e23a4f1f`，启动时工作树干净。
- 完成 50/50 epoch，无 early stopping、OOM、NaN 或 Inf；元数据状态为 `completed`。

## 2. Ultralytics 指标

| 指标 | P2 control | Variant A | 变化 |
|---|---:|---:|---:|
| 最佳 epoch | 40 | 50 | — |
| Precision | 0.53962 | 0.53243 | -0.00719 |
| Recall | 0.14626 | 0.11901 | -0.02725 |
| mAP50 | 0.14063 | 0.11684 | -0.02379 |
| mAP50-95 | 0.06737 | 0.05809 | -0.00928（-13.78%） |
| 有效训练时间 | 195.08 s | 206.42 s | +5.81% |

best checkpoint 独立验证得到 P 0.529、R 0.120、mAP50 0.117、mAP50-95 0.0581，与 CSV 最佳记录一致。

## 3. 尺寸专项评估

协议：固定 100 张 val，COCO 风格 101 点 AP，IoU 0.50:0.95，`conf=0.001`、`iou=0.7`、`maxDet=300`。

| 尺寸 | P2 control AP50-95 | Variant A AP50-95 | 绝对变化 | 相对变化 |
|---|---:|---:|---:|---:|
| all | 0.067716 | 0.057549 | -0.010167 | -15.01% |
| small | 0.016634 | 0.019108 | +0.002474 | +14.87% |
| medium | 0.105803 | 0.098278 | -0.007525 | -7.11% |
| large | 0.190013 | 0.164449 | -0.025564 | -13.45% |

| 尺寸 | P2 control AR50-95 | Variant A AR50-95 | 相对变化 |
|---|---:|---:|---:|
| all | 0.186840 | 0.180630 | -3.32% |
| small | 0.104344 | 0.095889 | -8.10% |
| medium | 0.271738 | 0.252142 | -7.21% |
| large | 0.299891 | 0.337344 | +12.49% |

Variant A 的 small AP 有恢复，但 small AR 仍下降，且整体、medium、large AP 明显退化，不满足“small 改善且整体无明显退化”的正式候选门槛。

## 4. 参数状态与判定

best checkpoint 中归一化全局权重约为 `[0.48618, 0.51377]`，gate weight 绝对值均值为 `0.01917`，两者均参与训练。相较完整 DWF，移除 DySample 将 small AP 从相对 control 的 -2.41% 改善为 +14.87%，但未消除整体负向，因此动态采样是负向因素之一而非唯一因素。

下一步按冻结路线执行 Variant B：`nearest + 仅全局权重`，去掉局部门控。如果 Variant B 仍不能同时改善 small 且保持整体，则淘汰 DWF 路线，恢复官方 P2 concat 并进入 SA-NWD。

## 5. 证据哈希

| 文件 | SHA256 |
|---|---|
| Variant A `results.csv` | `11bad8509748336ff79a1931f88758f3589fb3270c3cc455136c5c841274188f` |
| Variant A `best.pt` | `8873d57c410b88b59a92dbef1c475de8d57d9fe6a5f854d57ba60797b0314f5d` |
| Variant A 尺寸 JSON | `5ecc37a198abe47680a3a5b2a05f2d8a074d65d3ae96f327abbfda870869a437` |
