# HaarDownsample Stage20 筛选协议（拟P18方案A）

> 冻结日期：2026-08-27（Asia/Shanghai），在任何正式训练启动前写入。
> 状态：用户于2026-08-27授权启动本筛选；导师对创新类型口径的确认尚未完成，因此本阶段产物在获追认前只作为筛选级证据，不得直接进入论文表格。
> 来源候选：`docs/plans/MODULE_IMPROVEMENT_PROPOSALS.md`方案A。

## 1. 研究问题与假设

BDD道路小目标在640输入下的共同small漏检（52,563个）主要源于下采样信息损失。官方P2骨干最浅两处stride-2卷积（第0层P1/2、第1层P2/4）丢弃高频信息；用固定正交Haar DWT替换后，下采样变为无损变换加可学习子带融合，small AP应相对同协议官方P2控制组提高。

Haar DWT为固定数学变换（无学习参数），不存在核尺寸、衰减率、门宽度等变体搜索空间；子带融合卷积为单一3×3卷积+BN+SiLU，无变体。

## 2. 冻结设计

- 模块：`sdf/modules/haar_downsample.py::HaarDownsample(c1, c2, kernel_size=3)`，先`haar_dwt`（正交，能量保持）输出4·c1子带通道，再3×3融合卷积到c2通道。
- 插入点：`configs/models/yolo26n-p2-hwd.yaml`，仅替换官方`yolo26-p2.yaml`第0、1层，其余逐层相同（单元测试断言：深层权重与官方完全兼容，仅0/1层不兼容）。
- 正式配置：`configs/train/bdd100k-hwd-r1-candidate-640-e200-stop20-b32.yaml`——BDD 70k/10k、640、200轮计划、epoch20可恢复暂停、batch32/nbs64、seed0、deterministic、AMP。
- 预训练：`upstream/yolo26n.pt`，与控制组一致；第2层起全部加载，第0/1层随机初始化（与控制组的预训练不对称在结果解读中说明）。
- 设备：GPU1（device "1"，与另一长期任务共存）。控制组R0在GPU0训练；设备差异作为协议偏差记录。

## 3. 控制组：使用冻结R0，不重跑

对照为RDCI Stage20的R0官方P2控制组（同batch32/nbs64、seed0、200轮计划、epoch20暂停、BDD 10k尺寸评估），冻结值`results/rdci-stage20/summary.json`：

| 指标 | R0冻结值 |
|---|---|
| all AP | 0.235178 |
| small AP | 0.083358 |
| small AR | 0.257471 |
| medium AP | 0.267721 |
| large AP | 0.466453 |

不重跑R0的理由：协议逐字段一致（同一数据、调度、batch、seed、暂停边界、评估脚本与参数），重跑只消耗约3.5小时GPU而不改变比较性质。若R1意外大幅偏离R0（all AP差超过±5%相对值），须先排查工程故障再解读。

## 4. 执行步骤

1. [工程验收] 单元测试6项：DWT正交性/精确重构/奇数尺寸、模块前向反向/状态回环、CUDA FP16、YAML集成（2处模块、stride 4/8/16/32、深层权重兼容、参数增量0~30k）。
2. [工程smoke] `configs/train/bdd100k-hwd-p2-smoke-640.yaml`：固定500图子集、3 epoch、batch32。通过标准：无OOM、无NaN/Inf、无fitness恢复、显存峰值<20GB、参数与GFLOPs差落在预期区间（参数+15,120、GFLOPs约+4~5%）。
3. [正式Stage20] 启动器`scripts/run_experiment.py --config configs/train/bdd100k-hwd-r1-candidate-640-e200-stop20-b32.yaml`；epoch20完整可恢复暂停，checkpoint含optimizer/scaler/EMA。
4. [尺寸评估] `scripts/evaluate_object_sizes.py --model <R1 best.pt> --data configs/data/bdd100k.yaml --output results/haar-downsample-stage20/r1-size-val.json --device 1`（默认conf 0.001、IoU 0.7、640、val split，与R0评估一致）。
5. [收口] 结果、相对百分比、停止门判定写入`results/haar-downsample-stage20/summary.json`与报告；状态文档同步。

## 5. 预注册停止门（训练前冻结，事后不得调整）

以R1相对R0冻结值的百分比为准：

- **继续候选**：small AP ≥ +2.0% 且 all AP ≥ -0.5% 且 small AR ≥ 0（不退化）。记录为正向候选，是否进入Stage40/200轮须另行决策。
- **边缘记录**：small AP ∈ [+1.0%, +2.0%) 且 all AP ≥ -0.5%。记录为边缘，不进入Stage40。
- **关闭**：small AP < +1.0%，或 all AP < -0.5%，或 small AR退化。路线关闭，不追加变体、不调融合核尺寸、不换插入层、不叠加其他模块。

## 6. 禁止事项

- 不因结果调整门数值、融合核、插入位置或子带顺序后重跑。
- 不在同一份BDD val上做变体选择；本筛选是方案A的唯一一次预注册机会。
- 不恢复、不组合任何已关闭路线（SDD、DWF、SA-NWD、通道注意力、蒸馏、RDCI、PASCP、裁块PGSR）。
- 不访问VisDrone test-dev；不将本筛选结果写入论文，除非导师追认筛选协议。
- smoke指标只作工程健康证据，不作精度判断。
