# Phase 2 SDD 独立模块与固定 smoke 报告

> 工作包：`P2-SDD-PREP`  
> 日期：2026-07-28（Asia/Shanghai）  
> 结论：最小实现、单元/集成测试、导出和固定 smoke 通过；尚未形成精度结论。

## 1. 实现边界

- 实现 `SDD`（Small-object Detail-preserving Downsampling）：`PixelUnshuffle(2) → 1×1 Conv → 3×3 Depthwise Conv → 1×1 Conv → BN → SiLU`。
- 首版只替换官方 P2 backbone 中生成 P2/4 的第 2 处 stride-2 卷积；生成 P1、P3、P4、P5 的下采样保持不变。
- 不包含 DWF、SA-NWD 或其他自定义模块。
- 模块代码位于 `sdf/modules/sdd.py`；上游只在 `ultralytics/nn/tasks.py` 增加 YAML 注册点。
- 配置位于 `configs/models/yolo26n-p2-sdd.yaml`，可通过切换回官方 YAML 完整关闭 SDD。
- `sdf` 已通过 `pyproject.toml` 以 editable 模式安装，控制台入口和项目外路径均可加载自定义 checkpoint。

实现 commit：`e796a20`；可导入安装 commit：`e9ec5e8`。

## 2. 单元与集成测试

测试文件：`tests/test_sdd.py`。

| 测试 | 结果 |
|---|---|
| PixelUnshuffle 后 PixelShuffle 恢复原张量 | 通过，逐元素相等 |
| `16×64×96 → 32×32×48` shape | 通过 |
| FP32 前向/反向与全部梯度有限性 | 通过 |
| state_dict 保存、加载与输出一致性 | 通过 |
| 完整 YAML 构建且仅含 1 个 SDD | 通过 |
| 检测 stride `[4, 8, 16, 32]` | 通过 |
| CUDA FP16 autocast 反向与梯度有限性 | 通过 |
| 完整模型 TorchScript 导出和重新加载 | 通过，临时产物 10,588,180 bytes |
| 从项目外 `/tmp` 加载 smoke checkpoint | 通过 |
| `pip check` | 通过，无冲突 |

## 3. 复杂度与初始化

BDD100K 10 类、640 输入：

| 模型 | 参数量 | GFLOPs | stride |
|---|---:|---:|---|
| 官方 YOLO26n-P2 | 2,518,352 | 7.5 | `[4,8,16,32]` |
| P2 + SDD（仅 P2） | 2,517,104 | 7.5 | `[4,8,16,32]` |

SDD 自身为 3,424 参数；替换原卷积后整网净减少 1,248 参数，GFLOPs 在 Ultralytics 一位小数统计下不变。用 P2 `best.pt` 做工程 smoke 时迁移 901/904 项，新卷积权重需要重新学习。

## 4. 固定 BDD100K smoke

配置：`configs/train/sdd-p2-smoke-640.yaml`。

| 项目 | 结果 |
|---|---|
| 数据 | 固定 BDD100K smoke，400 train / 100 val |
| epoch / imgsz / batch | 3 / 640 / 64 |
| 设备 | 物理 GPU0，RTX 3090 24 GB |
| 初始化 | 已验收 P2 `best.pt`，仅用于工程连通性 |
| optimizer | AdamW（`optimizer=auto` 在短训练中的实际解析） |
| 最高日志显存 | 约 22.3 GB |
| 有效训练时间 | 13.30 秒 |
| train/val/save/final val | 全部正常退出 |
| NaN/Inf/OOM | 无 |
| 元数据状态 | `completed` |

训练 loss：

| epoch | box | cls | l1 | val box | val cls | val l1 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3.58778 | 5.54753 | 0.01013 | 3.51259 | 7.40860 | 0.01739 |
| 2 | 3.35561 | 4.35795 | 0.00894 | 3.45014 | 7.38100 | 0.01598 |
| 3 | 3.07439 | 3.50123 | 0.00713 | 3.47628 | 6.57276 | 0.01537 |

仅 3 epoch 且关键下采样层随机初始化，精度很低，不得用于论文或判断 SDD 有效性。本 smoke 只证明训练链路、梯度、AMP、显存、验证和检查点完整。

run：`runs/detect/phase2-smoke/bdd100k-yolo26n-p2-sdd-p2only-640-e3-b64-seed0/`。

## 5. 冻结证据

| 文件 | SHA256 |
|---|---|
| smoke 配置 | `59ff2ab4e2df31eea304fe80db2872c1204b36633e279efe7e2e1b5f03ceee07` |
| 数据 YAML | `30f3032f881bea875a72b1bffd55927a3acf54f26cb86225795a9b04f104aef4` |
| 模型 YAML | `b9c0ed120768b6029adda8c771f92fcca2fc3e7ddfa40e2de68e19803ae4f65d` |
| smoke 初始化权重 | `f1a8ad980cdaf949a12ac98514b3ccc583e1026fe4b09aa664e8d2589bc2826c` |
| smoke `best.pt` | `bc80e7af7f0477d610ead71204fb3d2a8c276c70e6a490ea0485fba82895d28e` |
| smoke `last.pt` | `2e44e161f3f334d5f30b4d59284bccd256ed1551e8a4f2352e6c0aabc894bffb` |

## 6. 下一步与公平性约束

下一步执行官方 P2 与 P2+SDD 的配对 50 epoch 短跑筛选。两者必须：

1. 从同一个原始 `upstream/yolo26n.pt` 初始化，不能让 SDD 正式候选从已经训练 200 epoch 的 P2 `best.pt` 起步。
2. 使用相同固定数据子集、seed、imgsz、batch、optimizer 和评估协议。
3. 分别记录整体和 small/medium/large 指标、参数量、GFLOPs、显存和耗时。
4. 短跑只用于淘汰和冻结结构，不进入论文正式主表。
5. 若 SDD 稳定且没有明显负向趋势，再冻结完整 BDD100K 200 epoch 配置；否则先检查“仅 P2”和“P2+P3”插入位置，不直接叠加 DWF。

## 7. 筛选结果同步

配对 50 epoch 筛选已于 2026-07-28 完成。最低风险的“仅 P2”SDD 在 all/small/medium/large AP50-95 上分别比同源 P2 对照低 49.78%/61.48%/60.27%/46.74%，满足淘汰条件。SDD 不进入完整 200 epoch 正式消融，也不叠加到后续模块；完整证据见 `results/reports/bdd-modules/p2_sdd_screen_report.md`。
