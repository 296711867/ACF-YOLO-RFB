# Phase 2 SA-NWD 工程与 smoke 验收报告

> 工作包：`P2-SA-NWD-PREP`  
> 日期：2026-07-29（Asia/Shanghai）  
> 结论：SA-NWD 最小实现、双分支接入和固定 smoke 通过，可以进入 P2 control/SA-NWD 配对 50 epoch 筛选。

## 1. 实现边界

活动网络保持已验收的官方 YOLO26n-P2，不包含 SDD 或 DWF。SA-NWD 只替换 bbox 回归中的单框 `1-CIoU` 项：

`(1 - alpha) * (1 - CIoU) + alpha * (1 - NWD)`。

YOLO26 原生 reg_max=1 的归一化 L1、分类 BCE、TaskAlignedAssigner、one-to-many/one-to-one 动态权重、数据和网络结构均保持不变。默认 `sa_nwd: false`，关闭时已验证 bbox loss 与原生 CIoU 逐位相等。

## 2. 首轮冻结参数

| 参数 | 数值 | 依据 |
|---|---:|---|
| NWD 常数 C | 32 px | 在映射到 640 尺度后平滑微小坐标偏移 |
| 面积衰减 τ | 1024 px² | 对齐 COCO small 的 32×32 面积边界 |
| alpha 上限 | 0.8 | 极小目标仍保留至少 20% CIoU |
| 参考尺度 | 640 | 640/960 等输入先映射到同一尺度计算面积与距离 |

alpha 在 640 参考尺度下：4×4 和 8×8 为 0.8，16×16 为 0.7788，32×32 为 0.3679，96×96 约 0.00012。

## 3. 工程测试

- SA-NWD 面积权重单调、上限、参数非法值和极小框数值边界通过。
- 640/960 等价框映射后的损失一致。
- stride 像素尺度转换、reg_max=1 L1 保留、前向/反向梯度有限。
- 默认关闭路径与原生加权 CIoU reduction 逐位一致。
- one-to-many 与 one-to-one 两套 `v8DetectionLoss` 均实例化相同 SA-NWD，原生 top-k 10/7 保持不变。
- SA-NWD、SDD、DWF 共 15 项回归测试通过；实际 RTX 3090 CUDA/FP16 autocast 损失与梯度有限。
- 新配置参数通过 Ultralytics 类型与范围校验。

## 4. 固定 smoke

配置：`configs/train/sa-nwd-p2-smoke-640.yaml`。

- 官方 P2、原始 `yolo26n.pt`、固定 BDD100K 400 train / 100 val、640、batch 32、seed 0、GPU0。
- 完成 3/3 epoch，无 early stopping、OOM、NaN 或 Inf；CSV 3 行全部有限。
- 最高日志显存约 12.5 GB；best/last、曲线、最终验证与元数据完整。
- 第 3 epoch 因极低早期 fitness 命中 Ultralytics 内置 `fitness == 0` 检查，自动从 last 恢复 1 次后成功完成。恢复前后 loss 均有限，checkpoint 权重通过有限性检查；该现象保留为筛选期监控项，不隐去。
- best checkpoint 的 `train_args` 保存了 `sa_nwd=true`、C、τ 和 alpha 上限；重载 stride 为 `[4, 8, 16, 32]`。

run：`runs/detect/phase2-smoke/bdd100k-yolo26n-p2-sa-nwd-640-e3-b32-seed0/`。

## 5. 筛选设计

重新运行一组同提交 P2 原生 CIoU control，再运行 SA-NWD candidate。两者除 `sa_nwd` 与 run name 外完全一致：400/100、640、50 epoch、batch 32、seed 0、GPU0。两组均执行 best 独立回验和尺寸评估；只有 small AP/AR 改善且整体无明显退化时才进入 200 epoch。

## 6. 证据哈希

| 文件 | SHA256 |
|---|---|
| smoke 配置 | `3549a6982ad81abb927f389152ada882cf21cce85a5114b88a30b59b1d6378f6` |
| control 配置 | `1a0d77cb9130a7040656c8cd51c18375a9259ced23f0b9c39a9c6023d61d7420` |
| candidate 配置 | `ac85208eb38aaa3df468ae069af7a9a8871e9e2877d01701f97ab488ee330a65` |
| SA-NWD 源码 | `6c476e6154e94e4a69741525e57f226d1d6e9c5fd156ce17a0dd2a70fa5c29d4` |
| smoke `results.csv` | `106461f21e37db38fe8608aa8a5ec3da61c422eb8d47049d7abf0936382d2c3f` |
| smoke best | `7176e452ba98df1ff9028ace85e3af781ed653a4fcaae91dc6f1bd44e57faa60` |
| smoke last | `624b845609a2137035417110e7a60a676bb30c47d77362a84086281b4e5a21b8` |
