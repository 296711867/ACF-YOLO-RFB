# Phase 2 DWF nearest+weight-only 诊断准备报告

> 工作包：`P2-DWF-DIAG-B-PREP`  
> 日期：2026-07-29（Asia/Shanghai）  
> 结论：Variant B 工程验收通过，可以进入与既有 batch 32 P2 control 配对的 50 epoch 筛选。

## 1. 诊断边界

Variant B 使用 nearest-neighbor 2× 上采样与归一化全局融合权重，移除 DySample 和局部门控；保留高低分辨率投影、轻量融合与官方 P2 其余网络。该变体不含 SDD/SA-NWD，目的仅是判断 Variant A 剩余退化是否来自局部门控。

## 2. 工程与 smoke 验收

- CPU 单元测试 6 项通过；实际 RTX 3090 CUDA/FP16 autocast 反向通过。
- YAML 中恰好 1 个 `DWFNearestWeight`、0 个 DySample、0 个 gate、0 个 SDD。
- shape、归一化权重、梯度和 checkpoint 重载通过，stride 为 `[4, 8, 16, 32]`。
- BDD 10 类模型 2,529,554 parameters、约 8.0 GFLOPs。
- 固定 BDD100K 400/100、640、batch 32、seed 0、GPU0 完成 3/3 epoch。
- 最高训练日志显存约 14.5 GB，无 early stopping、OOM、NaN 或 Inf。
- best/last、CSV、曲线、最终 best 验证和实验元数据完整；CSV 全部数值有限。

run：`runs/detect/phase2-smoke/bdd100k-yolo26n-p2-dwf-nearest-weight-640-e3-b32-seed0/`。

## 3. 证据哈希

| 文件 | SHA256 |
|---|---|
| smoke 配置 | `ff767c465ae457e4187f226446ea1a47ef86957eb7f0a98c49e5395f88d542a1` |
| screen 配置 | `e639fdf10a92c18c54e6bda52ca451da53254aafe60a383905f3d43e064d3313` |
| 模型 YAML | `61da554946a8b445137d209f943b9e26c93dfc36b68a79cd39379873fdfa7ed2` |
| smoke best | `70efe054cf133f4fa2abf367712d43c733257ab36430eb30799e300ca6d93d5c` |
| smoke last | `2c8de9fdf0919b0e0af3e791c917742267f2ff75abc3a2cbfb3e60a18461329a` |

## 4. 下一步

用 `configs/train/dwf-diag-nearest-weight-screen-640.yaml` 启动 50 epoch 筛选，与既有 P2 control 执行相同 best 回验和尺寸评估。若仍不能同时改善 small 且保持整体，则淘汰 DWF 并进入 SA-NWD。
