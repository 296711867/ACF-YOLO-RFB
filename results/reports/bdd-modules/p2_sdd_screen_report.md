# Phase 2 SDD 配对 50 epoch 短跑筛选报告

> 工作包：`P2-SDD-SCREEN`  
> 日期：2026-07-28（Asia/Shanghai）  
> 结论：SDD 候选淘汰，不进入完整 BDD100K 200 epoch 正式消融。

## 1. 公平性与边界

本筛选只比较以下两组：

| 组别 | 模型 | 自定义模块 |
|---|---|---|
| control | 官方 YOLO26n-P2 | 无 |
| candidate | 官方 P2 + 仅 P2 下采样 SDD | SDD |

除模型 YAML 和 run 名称外，两份配置完全一致：

- 同一 `upstream/yolo26n.pt` 初始化，SHA256 `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`。
- 同一固定 BDD100K 400 train / 100 val 子集、640、50 epoch、batch 64、seed 0、deterministic、AMP。
- 同一物理 GPU0、同一 `optimizer=auto`；短训练均解析为 AdamW。
- 不包含 DWF、SA-NWD 或数据增强差异。
- 筛选 commit：`06cacb073c1cf65659511cd6f33ebf786ff731df`。

配置哈希：

| 配置 | SHA256 |
|---|---|
| `configs/train/sdd-screen-p2-control-640.yaml` | `7aa5f9c816ebf7e475ae03c201d1f3a2fe18867fd7d1ac2ef082524fce95c8b5` |
| `configs/train/sdd-screen-p2-sdd-640.yaml` | `e479a1e23df576f0b6ad8b8d3bc50abc566ecd558ac9204771511a3a8f048528` |
| 固定数据 YAML | `30f3032f881bea875a72b1bffd55927a3acf54f26cb86225795a9b04f104aef4` |
| 官方 P2 YAML | `0dbc2fa3dfede5edcda2566ab8c7ea150ad066ea29d79e68d6c4da2bab476279` |
| P2+SDD YAML | `b9c0ed120768b6029adda8c771f92fcca2fc3e7ddfa40e2de68e19803ae4f65d` |

原始权重迁移：control 360/902 项，candidate 359/904 项。SDD 中三个卷积权重需要随机初始化，而 control 的对应浅层卷积可继承预训练权重。

## 2. 训练结果

两组均完成 50/50 epoch，无 early stopping、NaN、Inf 或 OOM；全部 CSV 数值有限，元数据状态均为 `completed`。

| 指标 | P2 control | P2+SDD | 变化 |
|---|---:|---:|---:|
| 最佳 epoch | 39 | 50 | — |
| Precision | 0.59366 | 0.36474 | -0.22892 |
| Recall | 0.14938 | 0.07681 | -0.07257 |
| mAP50 | 0.12640 | 0.06187 | -0.06453 |
| mAP50-95 | 0.05975 | 0.02918 | -0.03057（-51.16%） |
| 有效训练时间 | 215.42 s | 211.16 s | -1.98% |

最终 best checkpoint 独立回验 mAP50-95 分别约为 0.0592 和 0.0291，与 CSV 最佳记录一致。

训练趋势：

| epoch | P2 mAP50-95 | P2+SDD mAP50-95 |
|---:|---:|---:|
| 10 | 0.00059 | 0.00028 |
| 20 | 0.02645 | 0.00922 |
| 30 | 0.05030 | 0.02011 |
| 40 | 0.05789 | 0.02865 |
| 50 | 0.05903 | 0.02918 |

SDD 在整个有效学习阶段均落后；候选在 epoch 50 仍有缓慢提升，但差距约为 2 倍，不属于可忽略随机波动。

## 3. 尺寸专项评估

协议：固定 100 张 val，COCO 风格 101 点 AP，IoU 0.50:0.95，`conf=0.001`、`iou=0.7`、`maxDet=300`；面积阈值与正式实验一致。

| 尺寸 | P2 AP50-95 | P2+SDD AP50-95 | 绝对变化 | 相对变化 |
|---|---:|---:|---:|---:|
| all | 0.058125 | 0.029193 | -0.028932 | -49.78% |
| small | 0.016537 | 0.006370 | -0.010167 | -61.48% |
| medium | 0.090756 | 0.036054 | -0.054702 | -60.27% |
| large | 0.178872 | 0.095273 | -0.083599 | -46.74% |

| 尺寸 | P2 AR50-95 | P2+SDD AR50-95 | 相对变化 |
|---|---:|---:|---:|
| all | 0.177686 | 0.096924 | -45.45% |
| small | 0.105629 | 0.049080 | -53.54% |
| medium | 0.267035 | 0.147228 | -44.87% |
| large | 0.308442 | 0.197513 | -35.96% |

完整指标：

- `results/sdd-screen/p2-control-e50-size-metrics.json`
- `results/sdd-screen/p2-sdd-p2only-e50-size-metrics.json`

逐图预测各 100 行，保留在同目录并由 Git 忽略。

## 4. 判定与原因分析

SDD 同时显著降低整体、小、中、大目标 AP 和 AR，满足预先冻结的淘汰条件，因此：

1. 不启动 SDD 的完整 BDD100K 200 epoch 训练。
2. 不把 SDD 叠加到 DWF 或 SA-NWD，防止无效模块混入后续增益。
3. SDD 源码、配置、测试、日志和 checkpoint 保留，用于复查负结果；“淘汰”指不进入活动候选和最终网络，不删除导致历史 checkpoint 无法加载的代码。
4. 下一工作包基于已验收的官方 P2 开展 DWF 独立实现与 smoke。

一个合理但尚未被单独证明的解释是：SDD 替换了整个 backbone 上游的关键浅层卷积，其三个卷积随机初始化，短跑中早期特征恢复速度明显慢于可直接迁移该层的 control。由于最低风险的“仅 P2”插入已经失败，继续尝试 P2+P3 会扩大随机浅层替换和干预范围，不符合停止规则。

## 5. 文件哈希

| 文件 | SHA256 |
|---|---|
| control `results.csv` | `cf8ab138f33d2bd0f9e0247d55ba978bd092017757fdc8b05f00113eaf92fea1` |
| SDD `results.csv` | `efb9521e49da041c88147e6e29313e8b60c857af0163ddd343d607eeb8cbb56a` |
| control `best.pt` | `33f71f577a8936518cd6e27f15a3ceb617303bc8743f00e1880a0f9ad956779e` |
| SDD `best.pt` | `89905dcdf01aa25000b23c88582534062e6fc1a3f5d5df71a9480ce325cd1a37` |
| control 尺寸 JSON | `459942e6f9ec4613a7b309036dc9bf1ed0137ae0964589c499ac11fd3935ca27` |
| SDD 尺寸 JSON | `58abe10f5cf99c0a42c1015cdf1a35c2180aa1e74cd0b0d1470058f41abadeab` |
