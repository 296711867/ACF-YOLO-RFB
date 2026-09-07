# Phase 2 DWF 独立模块与固定 smoke 报告

> 工作包：`P2-DWF-PREP`  
> 日期：2026-07-29（Asia/Shanghai）  
> 结论：最小实现、单元/集成测试、导出和固定 smoke 通过；尚未形成精度结论。

## 1. 实现边界

- 上游 Ultralytics 8.4.104 没有可直接复用的 DySample 或等价动态融合模块，因此新增实现是必要的。
- DWF 只替换官方 P2 neck 的 P3→P2 融合；P5→P4、P4→P3 和 bottom-up 路径保持官方结构。
- 不包含已淘汰的 SDD，也不包含 SA-NWD。
- `DySample` 使用零初始化 offset；初始状态在 FP32 下与 `bilinear + align_corners=False` 的最大误差约 `4e-6`。
- 两路特征先投影到 64 通道；高层特征经可学习 offset 的 `grid_sample` 上采样，再经 BiFPN 式非负全局权重、单通道局部门控和 depthwise+pointwise 融合。
- YAML 中 layer 17 使用无计算 `Identity` 保持官方后续 layer index，DWF 位于原 Concat 的 layer 18，便于迁移后续 checkpoint 权重。
- 模块代码位于 `sdf/modules/dwf.py`；`upstream/ultralytics/nn/tasks.py` 只增加 DWF/SDD 名称和通道解析注册。

实现 commit：`0e72081e52663fd83f741d470c7566fbe5253d3c`。

## 2. 单元与集成测试

测试文件：`tests/test_dwf.py`。

| 测试 | 结果 |
|---|---|
| 零 offset 与 bilinear 上采样数值对齐 | 通过，容差 `1e-5` |
| 全局权重非负且归一化和约为 1 | 通过 |
| `[64,H,W] + [64,2H,2W] → [64,2H,2W]` | 通过 |
| FP32 前向、反向及全部梯度有限 | 通过 |
| CUDA FP16 autocast 反向及梯度有限 | 通过 |
| state_dict 保存、加载和输出一致 | 通过 |
| 完整 YAML 仅包含 1 个 DWF、0 个 SDD | 通过 |
| 检测 stride `[4,8,16,32]` | 通过 |
| 完整模型 TorchScript 导出和重新加载 | 通过，10,656,875 bytes |
| 项目外 `/tmp` 加载 smoke checkpoint | 通过 |

## 3. 复杂度与权重迁移

BDD100K 10 类、640 输入：

| 模型 | 参数量 | GFLOPs | stride |
|---|---:|---:|---|
| 官方 YOLO26n-P2 | 2,518,352 | 7.5 | `[4,8,16,32]` |
| P2 + DWF（仅 P3→P2） | 2,530,203 | 8.0 | `[4,8,16,32]` |

DWF 自身 13,899 参数；整网净增加 11,851 参数（约 0.47%），GFLOPs 约增加 6.7%。从已验收 P2 `best.pt` 可迁移 901/926 项，新增 DWF 参数随机初始化。

## 4. 固定 BDD100K smoke

配置：`configs/train/dwf-p2-smoke-640.yaml`。

| 项目 | 结果 |
|---|---|
| 数据 | 固定 BDD100K smoke，400 train / 100 val |
| epoch / imgsz / batch | 3 / 640 / 32 |
| 设备 | 物理 GPU0，RTX 3090 24 GB |
| 初始化 | 已验收 P2 `best.pt`，仅用于工程连通性 |
| optimizer | AdamW，lr 0.000714，momentum 0.9 |
| 最高日志显存 | 约 14.6 GB |
| 有效训练时间 | 12.94 秒 |
| train/val/save/final val | 全部正常退出 |
| NaN/Inf/OOM | 无 |
| 元数据状态 | `completed` |

训练记录：

| epoch | train box | train cls | train l1 | val box | val cls | val l1 | mAP50-95 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.97674 | 2.37922 | 0.00686 | 2.37113 | 2.97769 | 0.00596 | 0.17207 |
| 2 | 2.54232 | 1.93107 | 0.00477 | 2.33665 | 2.57222 | 0.00599 | 0.18089 |
| 3 | 2.36751 | 1.73319 | 0.00418 | 2.31324 | 2.28897 | 0.00591 | 0.18211 |

smoke 后全局权重约为 `[0.49766, 0.50229]`；offset weight 绝对值均值 `0.00210`，gate weight 绝对值均值 `0.00392`，证明动态采样和门控参数均获得更新。3 epoch 指标受 P2 正式权重初始化影响，只能证明工程链路，不得作为 DWF 精度结论。

run：`runs/detect/phase2-smoke/bdd100k-yolo26n-p2-dwf-p2only-640-e3-b32-seed0/`。

## 5. 确定性限制

PyTorch 2.6 明确警告 CUDA `grid_sampler_2d_backward` 没有 deterministic 实现。Ultralytics 使用 `torch.use_deterministic_algorithms(True, warn_only=True)`，因此训练可继续，但 DWF 不能承诺相同 seed 下逐位一致。

处理原则：

1. 固定数据、seed、设备、配置和软件版本。
2. 配对短跑在同一 GPU 上顺序执行。
3. 正式候选最终通过三随机种子均值和标准差报告不确定性。
4. 不把 `deterministic=True` 误写成 DWF CUDA 反向的位级确定性保证。

## 6. 冻结证据

| 文件 | SHA256 |
|---|---|
| smoke 配置 | `f97beebb2c1488f5fddbb09f1e5ed062123df87eb8b86e55f6210c419993a3af` |
| 数据 YAML | `30f3032f881bea875a72b1bffd55927a3acf54f26cb86225795a9b04f104aef4` |
| 模型 YAML | `9af7d52637d1d73a3c4f5fea1f7f7eef4ca0bdb3a45c3612c9e7fb4e2df86c2b` |
| smoke 初始化权重 | `f1a8ad980cdaf949a12ac98514b3ccc583e1026fe4b09aa664e8d2589bc2826c` |
| smoke `results.csv` | `d08e582a50275f9ac5711aabbb4cf79654a08ef8f539a441cb5b0265b4797f14` |
| smoke `best.pt` | `f42146279371505f4ecec7074ce4aa0dc8e28670e2ce04de70957fa28ee8687e` |
| smoke `last.pt` | `35b44c67fc485127f5b9d92d4a4bce2081d980bc4ca55377150c5bc267c8d5cd` |

## 7. 下一步

执行官方 P2 与 P2+DWF 的配对 50 epoch 短跑筛选：

- 两组从同一个原始 `upstream/yolo26n.pt` 初始化。
- 使用相同固定子集、imgsz 640、batch 32、seed 0、optimizer 和评估协议。
- 分别执行 best checkpoint 回验和 small/medium/large 专项评估。
- 短跑只用于淘汰，不进入论文正式主表。
- DWF 稳定且无明显负向趋势后，才冻结完整 BDD100K 200 epoch 配置。

## 8. 筛选结果同步

完整 DWF 配对 50 epoch 筛选已于 2026-07-29 完成。相对同源 P2 control，all/small/medium/large AP50-95 分别变化 -19.78%/-2.41%/-11.47%/-23.73%，不满足正式训练门槛。当前完整 DWF 不进入 200 epoch；按预定降级路线继续测试 nearest+gate 和 nearest+weight-only 两个诊断变体。完整证据见 `results/reports/bdd-modules/p2_dwf_screen_report.md`。
