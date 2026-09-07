# 非对称容量重分配（P18B ACR）Stage20 筛选协议

> 冻结日期：2026-08-28（Asia/Shanghai），在正式训练启动前写入。
> 状态：用户授权启动，计划2026-08-28晚间执行；导师追认前结果为筛选级证据，不进入论文表格。
> 来源：`docs/plans/MODULE_IMPROVEMENT_PROPOSALS.md`补充的方案G（2026-08-28结构分析推论）。

## 1. 研究问题与假设

官方P2结构存在容量倒挂：small目标（<32²px）被分配到P2检测分支，但该分支检测特征仅32通道（L19），而最易检测的P5大目标分支有256通道。P18A已证明"无损保存信息"不解决问题（Haar DWT使small AR +0.58%但small AP -0.89%）：在nano宽度下瓶颈不是信息量，而是小目标分支的表示容量。

假设：将P2检测分支容量翻倍至与P3一致（64通道），同时把P5头分支减配（256→192通道）作为代偿，small AP相对同协议官方P2控制组显著提升，且因为属"信息通路/容量"家族而非"特征加权"家族，应突破历史±2%的加权家族天花板。

与已失败路线的机制区别（预注册，防止事后混淆）：浅层通道注意力（200轮未过门）是在**不增加容量**的前提下对32通道做重加权；ACR直接改变到达检测头的**通道容量**，属P2分支本身+14.3%所在的家族。

## 2. 冻结设计

- 模型：`configs/models/yolo26n-p2-acr.yaml`，相对官方`yolo26-p2.yaml`仅两处宽度变化：
  - L19（P2检测分支C3k2）输出32→64通道（标称128→256）；
  - L28（P5头分支C3k2）输出256→192通道（标称1024→768）。
  - 连带：L20输入64通道（输出仍32）、Detect头P2/P5 stem宽度。其余逐层与官方相同（单元测试断言）。
- 预训练兼容：backbone L0-L18全部官方权重按名字+形状加载；L19/L20/L28/L29随机初始化（测试断言兼容边界）。相比P18A只有两层随机初始化、且不涉及backbone深层语义。
- 成本（nc=80实测，thop同工具）：参数2,662,400→2,490,624（**-171,776，-6.45%**，净减）；GMACs 4.765→5.579（**+0.814，+17.08%**，P2在160×160翻倍的高分辨率代价，P5在20×20减配无法抵消）。此成本为设计固有，非偏差，已预注册披露。
- 正式配置：`configs/train/bdd100k-acr-r1-candidate-640-e200-stop20-b32.yaml`——BDD 70k/10k、640、200轮计划、epoch20可恢复暂停、batch32/nbs64、seed0、deterministic、AMP、GPU1（device "1"）。

## 3. 控制组：使用冻结R0，不重跑

对照为RDCI Stage20的R0官方P2控制组（协议逐字段一致），冻结值`results/rdci-stage20/summary.json`：all AP 0.235178、small AP 0.083358、small AR 0.257471、medium AP 0.267721、large AP 0.466453。若R1在线终点相对R0偏差异常（>±5%），先排查工程故障再解读。

## 4. 执行步骤（晚间按序执行）

1. [已由代理完成] 5项单元测试通过；全量测试与审计在提交前执行。
2. [工程smoke] `scripts/run_experiment.py --config configs/train/bdd100k-acr-p2-smoke-640.yaml`：500图3 epoch。通过标准：无OOM、无NaN/Inf、无fitness恢复、显存峰值<20GB。
3. [正式Stage20] `scripts/run_experiment.py --config configs/train/bdd100k-acr-r1-candidate-640-e200-stop20-b32.yaml`（约4小时，GPU1与RL任务共存；训练期间不跑任何计时基准）。
4. [尺寸评估] `scripts/evaluate_object_sizes.py --model runs/detect/phase18b-acr-stage20/bdd100k-yolo26n-p2-acr-r1-candidate-640-e200-stop20-b32-seed0/weights/best.pt --data configs/data/bdd100k.yaml --output results/acr-stage20/r1-size-val.json --device 1`（conf 0.001、IoU 0.7，与R0一致）。
5. [收口] 结果、相对百分比、门判定写入`results/acr-stage20/summary.json`与报告，同步状态文档。

## 5. 预注册停止门（训练前冻结，事后不得调整）

以R1相对R0冻结值的百分比为准：

- **继续候选**：small AP ≥ +3.0% 且 all AP ≥ -1.0% 且 small AR ≥ 0。理由：容量/通路家族是历史上唯一有两位数收益的家族（P2分支+14.3%）；真正的容量修复应明确超过加权家族±2%的天花板，故取+3%。
- **边缘记录**：small AP ∈ [+1.5%, +3.0%) 且 all AP ≥ -1.0%。记录但不进入Stage40。
- **关闭**：small AP < +1.5%，或 all AP < -1.0%（P5减配或P2误检放大致总体退化），或 small AR 退化。
- 附带观察项（不设门，只记录）：large AP 相对变化（检验P5减配代价）、误检数量变化（对照误差分析12,948误检基数）。

## 6. 禁止事项

- 不因结果调整门数值、L19/L28宽度比或任何其它层宽度后重跑。
- 不在同一份BDD val上做变体选择；本筛选是ACR的唯一一次预注册机会（宽度扫描64/96、P5减配比例搜索均属变体，禁止）。
- 不恢复、不组合已关闭路线；不访问VisDrone test-dev。
- smoke指标只作工程健康证据；在线mAP曲线不作为停止门依据（P18A已证在线终点与尺寸评估结论可背离）。
- 结果未获导师追认前不进入论文表格。
