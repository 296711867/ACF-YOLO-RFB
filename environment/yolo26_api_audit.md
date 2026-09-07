# YOLO26 接口审计

审计日期：2026-07-29（Asia/Shanghai）

## 来源与冻结信息

- 源目录：`<HOME>/code/github-project/yolo26/ultralytics-main`
- 冻结位置：`upstream/`
- 上游包版本：Ultralytics `8.4.104`
- 源目录没有 `.git`，无法恢复仓库 URL、tag 或 commit；不得把版本号误写成 Git commit。
- 原始冻结快照文件树 SHA256：`2faef3576f47aed79ff6ddaec511825d37b4148fe9046396347bd0322ee54f70`
- 当前工作树 SHA256：`ec1e350001783f7c436c76682b9482bf6e80457aaf23c580749ea9aa520b53f1`；变化为项目自定义模块注册和 SA-NWD 损失接入，见下文。
- 权重：`upstream/yolo26n.pt`
- 权重 SHA256：`9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`

文件树哈希生成方式：

```bash
find upstream -type f -not -path '*/__pycache__/*' -print0 | sort -z | xargs -0 sha256sum | sha256sum
```

## 模型接口

- YOLO26 检测 YAML：`upstream/ultralytics/cfg/models/26/yolo26.yaml`
- 官方 P2 YAML：`upstream/ultralytics/cfg/models/26/yolo26-p2.yaml`
- 模型构建：`upstream/ultralytics/nn/tasks.py` 中的 `DetectionModel` 和 `parse_model`。
- 检测头：`upstream/ultralytics/nn/modules/head.py:38` 的 `Detect`。
- 基线输出层：P3/8、P4/16、P5/32；实测 stride 为 `[8, 16, 32]`。
- 官方 P2 输出层：P2/4、P3/8、P4/16、P5/32；实测 stride 为 `[4, 8, 16, 32]`。
- 基线参数量：2,572,280；官方 P2 参数量：2,662,400。
- 原生 `end2end: true`，为端到端/NMS-free 检测路径。
- 原生 `reg_max: 1`，因此当前 bbox 分支使用归一化 L1 项而不是传统多 bin DFL。

结论：P2 已由官方提供，项目直接复用官方配置作为 P2 消融起点，不再实现一份重复分支。

## 项目扩展注册

- 自定义实现保存在顶层 `sdf/`，没有把模块主体复制进 Ultralytics layer zoo。
- `upstream/ultralytics/nn/tasks.py` 的项目差异仅包括：导入 `DWF/SDD`、把 SDD 纳入 base-module 通道缩放、为双输入 DWF 解析输入与输出通道。
- SDD 注册 commit：`e796a20`；DWF 注册 commit：`0e72081`。
- 自定义 YAML 分别为 `configs/models/yolo26n-p2-sdd.yaml` 和 `configs/models/yolo26n-p2-dwf.yaml`。
- SDD 已在配对短跑中淘汰，注册仅为历史 checkpoint 可加载性保留。
- DWF 模型保持原生 Detect 接口和 stride `[4,8,16,32]`，TorchScript 导出与重载通过。

## 损失与标签分配

- 标准检测损失：`upstream/ultralytics/utils/loss.py:337` 的 `v8DetectionLoss`。
- bbox 回归：`upstream/ultralytics/utils/loss.py:111` 的 `BboxLoss`。
- 标签分配：`upstream/ultralytics/utils/tal.py:14` 的 `TaskAlignedAssigner`。
- 端到端组合：`upstream/ultralytics/utils/loss.py:1280` 的 `E2ELoss`。
- criterion 选择：`upstream/ultralytics/nn/tasks.py:597`；YOLO26 的 end-to-end 模式选择 `E2ELoss`。
- `E2ELoss` 同时维护 one-to-many 和 one-to-one 两条损失，并对权重做训练期衰减。

SA-NWD 已按该接口实现：`sdf/losses/sa_nwd.py` 保存公式主体，`BboxLoss` 仅在开关开启时把单框 `1-CIoU` 换成尺寸感知混合项。`E2ELoss` 的 one-to-many/one-to-one 两条分支一致实例化；原生 reg_max=1 L1、分类损失、标签分配器和动态权重机制保持不变。默认开关关闭并通过原生 CIoU 逐位等价测试。

## 默认训练与导出接口

- 默认输入：640。
- 默认优化器：`auto`；实际解析结果取决于训练参数，正式 run 必须从日志读取并保存。
- 默认 `lr0=0.01`、`lrf=0.01`、`box=7.5`、`cls=0.5`、`dfl=1.5`。
- 默认增强：`mosaic=1.0`、`mixup=0.0`、`copy_paste=0.0`。
- Python 导出入口：`YOLO(...).export(...)`，实现位于 `upstream/ultralytics/engine/model.py:720`。
- 当前已验证自定义模型 TorchScript 导出；ONNX/TensorRT 仍不属于当前实现边界。

## 已完成验证

- editable import 指向本项目 `upstream/ultralytics/__init__.py`。
- 基线 YAML 构建通过。
- 官方 P2 YAML 构建通过。
- `yolo26n.pt` 加载通过。
- RTX 3090 GPU 0 本地图片预测通过：5 个检测，类别 ID 为 0 和 5。
- `pip check`：无依赖冲突。

## 已知限制

- 源快照缺少 Git 元数据，当前证据是版本号、权重哈希和文件树哈希，不等价于官方 commit。
- BDD100K 与 VisDrone 已下载、转为 YOLO 格式并通过 Ultralytics 全量扫描；BDD100K 基线与官方 P2 正式实验均已完成。
- SDD 已完成工程测试但因配对短跑显著退化而淘汰。
- DWF 及两个降级诊断均未通过配对短跑，路线已淘汰；CUDA `grid_sample` 限制仅影响保留的历史负结果。
- SA-NWD 已完成连续 τ=1024、small-only、连续 τ=512 三组配对短跑，均因整体 AP 退化而淘汰；代码默认关闭并仅为负结果复查保留。
