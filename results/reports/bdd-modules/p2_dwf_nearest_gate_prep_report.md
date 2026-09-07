# Phase 2 DWF nearest+gate 诊断准备报告

> 工作包：`P2-DWF-DIAG-A-PREP`  
> 日期：2026-07-29（Asia/Shanghai）  
> 结论：Variant A 工程验收通过，可以进入与既有 batch 32 P2 control 配对的 50 epoch 筛选。

## 1. 诊断边界

Variant A 只把完整 DWF 的 `DySample` 替换为无参数的 nearest-neighbor 2× 上采样，保留：

- 高、低分辨率特征投影；
- 归一化全局融合权重；
- 局部门控；
- 深度可分离轻量融合；
- 官方 P2 的其余骨干、颈部、四尺度检测头和损失。

该变体不包含 SDD 或 SA-NWD，目的仅是隔离动态采样是否为完整 DWF 退化的主要来源。

## 2. 工程测试

- CPU 单元测试：5 项通过，CUDA 专项在无设备上下文中跳过。
- 实际 RTX 3090 CUDA/FP16 autocast 反向：通过，输入、参数梯度均有限。
- YAML 构建：恰好 1 个 `DWFNearestGate`、0 个 `DySample`、0 个 SDD。
- 检测 stride：`[4, 8, 16, 32]`。
- shape、归一化权重、反向传播和 checkpoint 重载通过。
- BDD 10 类模型：2,529,683 parameters、约 8.0 GFLOPs。

## 3. 固定 smoke

配置：`configs/train/dwf-diag-nearest-gate-smoke-640.yaml`。

- 固定 BDD100K 400 train / 100 val，640，batch 32，seed 0，GPU0。
- 完成 3/3 epoch，无 early stopping、OOM、NaN 或 Inf。
- 最高训练日志显存约 14.7 GB。
- `results.csv` 共 3 行，所有数值有限。
- best/last、曲线、最终 best 验证和实验元数据完整。
- best checkpoint 项目外重载及四尺度 stride 复查通过。

run：`runs/detect/phase2-smoke/bdd100k-yolo26n-p2-dwf-nearest-gate-640-e3-b32-seed0/`。

## 4. 证据哈希

| 文件 | SHA256 |
|---|---|
| smoke 配置 | `c774f6fa5943cc4bb8f423b3056097088f84efe9181afc1292fff7adcc99be36` |
| screen 配置 | `17e1bba5e7ff1ebb8de34ec140f93b0d926dccf65693d46f584a53b5d9fd3068` |
| 模型 YAML | `fb28da3407c48882e2e00e8def0f3e2c9defe3bb5f2c3413b5b04d6a59c48bb9` |
| 预训练权重 | `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef` |
| smoke best | `aeefd526233d5a93172faa51cbc6dbc792a80868aaf54d8ac939f117c7e6b6f7` |
| smoke last | `3a061c9ccc3b7ded41a38c73272954fc96e2ae63d76ff356382a3765f4ecd7e5` |

## 5. 下一步

使用 `configs/train/dwf-diag-nearest-gate-screen-640.yaml` 启动 50 epoch 筛选，并与既有 control
`bdd100k-yolo26n-p2-control-dwf-pair-640-e50-b32-seed0` 比较。完成后必须回验 best checkpoint，并执行相同协议的小/中/大目标评估；短跑结果不进入论文正式主表。
