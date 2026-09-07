# 官方 YOLO26-P2 独立消融准备报告

> 工作包：`P1-OFFICIAL-P2`
> 日期：2026-07-27（Asia/Shanghai）
> 结论：官方结构、batch 64 smoke、正式训练与独立评估均通过；最终结论见 `results/reports/baselines/phase1_official_p2_640_report.md`。

## 1. 结构边界

- 模型直接使用 `upstream/ultralytics/cfg/models/26/yolo26-p2.yaml`。
- 检测尺度为 P2/P3/P4/P5，stride `[4, 8, 16, 32]`。
- 不包含 SDD、DWF、SA-NWD 或任何自定义模块。
- 初始化权重与基线相同，使用 `upstream/yolo26n.pt`。
- 官方 P2 因增加高分辨率分支，仅能迁移 360/902 个匹配项；新增层随机初始化，该事实已进入正式元数据。

BDD100K 10 类实例化后的复杂度：

| 模型 | 参数量 | GFLOPs | 输出尺度 |
|---|---:|---:|---|
| YOLO26n baseline | 2,507,700 | 5.8 | P3/P4/P5 |
| YOLO26n-P2 | 2,518,352 | 7.5 | P2/P3/P4/P5 |

P2 参数增加约 0.42%，GFLOPs 增加约 29.3%。

## 2. 固定 smoke

配置：`configs/train/p2-smoke-640.yaml`。

| 项目 | 结果 |
|---|---|
| 数据 | BDD100K smoke，400 train / 100 val |
| epoch | 1 |
| imgsz / batch | 640 / 64 |
| 设备 | 物理 GPU0，RTX 3090 24 GB |
| 峰值显存 | 20.8 GB |
| optimizer | AdamW，lr 0.000714，momentum 0.9 |
| 训练/验证 | 正常退出 |
| NaN/Inf | 无 |
| 产物 | args、CSV、best/last、曲线完整 |

smoke 仅验证工程和显存，1 epoch 精度禁止用于论文对比。峰值显存约占可见 24.1 GB 的 86%，剩余约 3.3 GB；固定 batch 64，以保持和基线一致。

smoke run：`runs/detect/phase1-smoke/bdd100k-yolo26n-p2-640-e1-b64-seed0/`。

## 3. 正式配置

配置：`configs/train/bdd100k-p2-640.yaml`。

- BDD100K 70k/10k。
- 640、200 epoch、batch 64、seed 0、deterministic。
- 物理 GPU0，显式 `CUDA_VISIBLE_DEVICES=0`。
- `optimizer=auto`；正式总迭代数超过 10,000，预期解析为 MuSGD，以实际日志为准。
- run：`runs/detect/phase1/bdd100k-yolo26n-p2-640-e200-seed0/`。
- 启动器记录模型 YAML、预训练权重、数据和配置的独立 SHA256。
- 已有 run 时拒绝覆盖，长训练必须置于独立 tmux 会话。

## 4. 验收门槛

- [x] 官方 YAML 构建和 stride 检查通过。
- [x] 同源预训练权重加载通过并记录迁移比例。
- [x] batch 64 前向、反向、val、保存通过。
- [x] 峰值显存低于 GPU0 可见显存。
- [x] 正式配置与基线保持数据、分辨率、epoch、batch 和 seed 一致。
- [x] 正式 200 epoch 已完成，无 early stopping、NaN、Inf 或 OOM。
- [x] 独立 val 和尺寸专项评估已完成；small AP50-95 相对基线提升 14.32%。

## 5. 正式启动记录

- 启动时间：2026-07-27 08:43（Asia/Shanghai）。
- 训练 commit：`d941b9de75666a521ceb0a425e00694762020002`。
- tmux：`yolo26-p2-640`。
- 物理映射：`CUDA_VISIBLE_DEVICES=0`。
- 正式 optimizer：MuSGD，lr 0.01，momentum 0.9。
- epoch 1 实测显存约 21.4 GB，GPU0 利用率约 91%，无 OOM/NaN/Inf。

## 6. 完成记录

- 训练于 2026-07-28 19:46 正常完成 200/200 epoch，最佳 epoch 183。
- 独立 val：P 0.674、R 0.434、mAP50 0.477、mAP50-95 0.269。
- 尺寸 AP50-95：small 0.101480、medium 0.321219、large 0.519476。
- P2 验收通过并保留；完整证据和成本分析见 `results/reports/baselines/phase1_official_p2_640_report.md`。
