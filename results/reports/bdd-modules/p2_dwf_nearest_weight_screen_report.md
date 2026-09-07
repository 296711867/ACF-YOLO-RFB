# Phase 2 DWF Variant B 配对 50 epoch 筛选与路线终止报告

> 工作包：`P2-DWF-DIAG-B-SCREEN`  
> 日期：2026-07-29（Asia/Shanghai）  
> 结论：nearest+仅全局权重仍未通过；DWF 路线正式淘汰，活动网络恢复官方 P2，下一阶段进入 SA-NWD。

## 1. 公平性与完成状态

- 与既有 batch 32 P2 control 使用相同原始权重、固定 BDD100K 400/100、640、50 epoch、seed 0 和 GPU0。
- 仅在 P3→P2 使用 nearest、归一化全局权重和轻量融合，不含 DySample、局部门控、SDD 或 SA-NWD。
- 启动提交：`3056c32481bbaac27fb70bb735b91e40e0467935`，启动时工作树干净。
- 完成 50/50 epoch，无 early stopping、OOM、NaN 或 Inf；元数据状态为 `completed`。

## 2. Ultralytics 指标

| 指标 | P2 control | Variant B | 变化 |
|---|---:|---:|---:|
| 最佳 epoch | 40 | 41 | — |
| Precision | 0.53962 | 0.38996 | -0.14966 |
| Recall | 0.14626 | 0.13337 | -0.01289 |
| mAP50 | 0.14063 | 0.12627 | -0.01436 |
| mAP50-95 | 0.06737 | 0.05551 | -0.01186（-17.61%） |
| 有效训练时间 | 195.08 s | 202.73 s | +3.92% |

best checkpoint 独立验证得到 P 0.391、R 0.134、mAP50 0.119、mAP50-95 0.0532；验证参数固定为 `conf=0.001`、`iou=0.7`、`maxDet=300`。

## 3. 尺寸专项评估

| 尺寸 | P2 control AP50-95 | Variant B AP50-95 | 绝对变化 | 相对变化 |
|---|---:|---:|---:|---:|
| all | 0.067716 | 0.055927 | -0.011789 | -17.41% |
| small | 0.016634 | 0.014571 | -0.002063 | -12.40% |
| medium | 0.105803 | 0.101834 | -0.003969 | -3.75% |
| large | 0.190013 | 0.157435 | -0.032578 | -17.15% |

| 尺寸 | P2 control AR50-95 | Variant B AR50-95 | 相对变化 |
|---|---:|---:|---:|
| all | 0.186840 | 0.182082 | -2.55% |
| small | 0.104344 | 0.088265 | -15.41% |
| medium | 0.271738 | 0.282977 | +4.14% |
| large | 0.299891 | 0.294252 | -1.88% |

Variant B 没有 small AP/AR 收益且整体明显退化，未通过正式候选门槛。

## 4. DWF 因果诊断总结

| 变体 | 相对 control all AP | 相对 control small AP | 判定 |
|---|---:|---:|---|
| 完整 DWF：DySample+weight+gate | -19.78% | -2.41% | 失败 |
| Variant A：nearest+weight+gate | -15.01% | +14.87% | small 恢复但整体失败 |
| Variant B：nearest+weight-only | -17.41% | -12.40% | 失败 |

移除 DySample 可恢复部分 small AP，说明动态采样是负向因素之一；继续移除 gate 后 small AP 再次下降，说明局部门控并非整体退化的唯一来源，并可能对小目标有局部帮助。剩余的加权求和/投影式融合仍不能替代官方 P2 的 `nearest + concat + C3`。因此不再扩展 DWF 变体、不做 200 epoch、不与 SA-NWD 叠加，活动网络恢复官方 P2。

best checkpoint 中归一化全局权重约 `[0.46844, 0.53151]`，确认权重发生更新。

## 5. 下一步

以已验收的官方 YOLO26n-P2 为唯一网络基座，开发尺寸感知混合 NWD 回归损失；先保持分类损失、标签分配器、网络结构和数据不变，并同时兼容 one-to-many/one-to-one 损失分支。完成损失单元测试和固定 smoke 后，再做同协议配对短跑。

## 6. 证据哈希

| 文件 | SHA256 |
|---|---|
| Variant B `results.csv` | `b54674210c0a7f7642754c4b76fd5374889b1ed1b66ce0694c03648695e6bf5a` |
| Variant B `best.pt` | `850ac0668c35ec51bc12f84be2ef0b8caa1100d7d016bc237d8b0357aad40f38` |
| Variant B 尺寸 JSON | `1310ab4ec01385a36de6689680272c5a433d463fdbc8290b3509c0c11efc4ff2` |
