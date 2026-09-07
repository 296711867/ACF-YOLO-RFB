# Phase 2 DWF 配对 50 epoch 短跑筛选报告

> 工作包：`P2-DWF-SCREEN`  
> 日期：2026-07-29（Asia/Shanghai）  
> 结论：当前完整 DWF 候选筛选失败，不进入 200 epoch；按预定降级路线继续做 nearest/无门控诊断。

## 1. 配对公平性

| 组别 | 模型 | 差异 |
|---|---|---|
| control | 官方 YOLO26n-P2 | 原生 nearest + concat |
| candidate | 官方 P2 + DWF | P3→P2 动态采样、全局权重、局部门控、轻量融合 |

两份配置除 model/name 外完全一致：

- 同一 `upstream/yolo26n.pt`，SHA256 `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`。
- 同一固定 BDD100K 400 train / 100 val、640、50 epoch、batch 32、seed 0、AMP、`optimizer=auto`。
- 同一物理 GPU0 顺序执行；`nbs=64`，两组有效 batch 策略一致。
- 不包含 SDD 或 SA-NWD。
- 两组从原始权重均迁移 360 项，DWF 新参数随机初始化。
- 筛选 commit：`136a0216727bee37a0380d2cc1fcdbefa4914b5d`。

| 配置 | SHA256 |
|---|---|
| `configs/train/dwf-screen-p2-control-640.yaml` | `737630a9e730a635d4365e3091b2fa17a3a03ea4536398aa141f6492d9e545a1` |
| `configs/train/dwf-screen-p2-dwf-640.yaml` | `bd910d9b4f06b865b012b14a6921b70ab42a801ce4f4f8b34773ae66e37efce5` |
| 固定数据 YAML | `30f3032f881bea875a72b1bffd55927a3acf54f26cb86225795a9b04f104aef4` |
| 官方 P2 YAML | `0dbc2fa3dfede5edcda2566ab8c7ea150ad066ea29d79e68d6c4da2bab476279` |
| P2+DWF YAML | `9af7d52637d1d73a3c4f5fea1f7f7eef4ca0bdb3a45c3612c9e7fb4e2df86c2b` |

## 2. 训练结果

两组均完成 50/50 epoch，无 early stopping、NaN、Inf 或 OOM；全部 CSV 数值有限，元数据状态为 `completed`。DWF 训练按预期出现 CUDA grid-sample backward 非位级确定性警告。

| 指标 | P2 control | P2+DWF | 变化 |
|---|---:|---:|---:|
| 最佳 epoch | 40 | 44 | — |
| Precision | 0.53962 | 0.52889 | -0.01073 |
| Recall | 0.14626 | 0.12898 | -0.01728 |
| mAP50 | 0.14063 | 0.12303 | -0.01760 |
| mAP50-95 | 0.06737 | 0.05820 | -0.00917（-13.61%） |
| 有效训练时间 | 195.08 s | 204.99 s | +5.08% |

训练趋势：

| epoch | P2 mAP50-95 | P2+DWF mAP50-95 |
|---:|---:|---:|
| 10 | 0.01007 | 0.01498 |
| 20 | 0.04104 | 0.03519 |
| 30 | 0.06056 | 0.05111 |
| 40 | 0.06737 | 0.05680 |
| 50 | 0.06384 | 0.05682 |

DWF 在早期 epoch 10 短暂领先，但从 epoch 20 起持续落后；最佳点也未追平 control。

## 3. 尺寸专项评估

协议：固定 100 张 val，COCO 风格 101 点 AP，IoU 0.50:0.95，`conf=0.001`、`iou=0.7`、`maxDet=300`。

| 尺寸 | P2 AP50-95 | P2+DWF AP50-95 | 绝对变化 | 相对变化 |
|---|---:|---:|---:|---:|
| all | 0.067716 | 0.054324 | -0.013392 | -19.78% |
| small | 0.016634 | 0.016233 | -0.000401 | -2.41% |
| medium | 0.105803 | 0.093667 | -0.012136 | -11.47% |
| large | 0.190013 | 0.144928 | -0.045085 | -23.73% |

| 尺寸 | P2 AR50-95 | P2+DWF AR50-95 | 相对变化 |
|---|---:|---:|---:|
| all | 0.186840 | 0.164856 | -11.77% |
| small | 0.104344 | 0.097562 | -6.50% |
| medium | 0.271738 | 0.242370 | -10.81% |
| large | 0.299891 | 0.297553 | -0.78% |

DWF 没有提高 small AP/AR，同时明显损害整体、medium 和 large AP，因此不满足正式训练门槛。

完整指标：

- `results/dwf-screen/p2-control-e50-b32-size-metrics.json`
- `results/dwf-screen/p2-dwf-p2only-e50-b32-size-metrics.json`

## 4. 动态参数状态

候选 best checkpoint 中：

- 归一化全局权重约 `[0.48911, 0.51084]`。
- offset weight 绝对值均值 `0.00860`。
- gate weight 绝对值均值 `0.02039`。

动态采样和门控均获得更新，负结果不是模块未参与训练导致。

## 5. 判定与下一诊断

当前“DySample + 全局权重 + 局部门控”完整 DWF 候选筛选失败，不启动 200 epoch。按照方案第 11.2 节，下一步依次诊断：

1. `nearest + 全局权重 + 局部门控`：隔离 DySample 是否是主要负向来源。
2. `nearest + 仅全局权重`：去掉局部门控，判断门控是否导致退化。
3. 两个变体仍无效时，淘汰 DWF，恢复官方 P2 nearest+concat，不继续组合 SA-NWD。

这些诊断仍只保留 P3→P2 一个融合点，并与 batch 32 P2 control 使用相同 50 epoch 协议。当前完整 DWF 代码和产物保留以复查负结果。

## 6. 文件哈希

| 文件 | SHA256 |
|---|---|
| control `results.csv` | `a0f42128cfe1fd7c2d1d6d106dc8599f7080c53b61861564481a0ca8967ebb42` |
| DWF `results.csv` | `2f1a294325de1f43293f090e13136b61c9779eb3703f25fd5c5d948b1605fb22` |
| control `best.pt` | `9c90688f7e93c9c599b998cb4ce388620aa77eae04df11df49dba5fb32cd10fe` |
| DWF `best.pt` | `42ae8aae362b7e4c51ff2351447d08d562006457ea4f04494857fbbe815b64e4` |
| control 尺寸 JSON | `cf848b3e2e7ceb37b273f7368b116e8bce4cae0f88bf867911cdc3b7973d4b37` |
| DWF 尺寸 JSON | `9338bd530f6cdae0d26f62f5dcdef7cae27f335a613989cedc8d618d5c836254` |
