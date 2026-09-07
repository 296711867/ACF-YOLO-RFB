# 容量对齐对照（P18C widen-only）协议

> 冻结日期：2026-09-01（Asia/Shanghai），训练启动前写入。
> 状态：确认性消融，非选型变体；导师追认前为筛选级证据。

> **2026-09-01启动前修订（用户授权）**：原设计为Stage20筛选（4小时）必要时再续200轮。用户决定直接全训200轮（约31小时，GPU空闲），判定锚点由20轮冻结值改为200轮冻结值（R0-200 small AP 0.103266、ACR-200 0.110785），使论文消融表三列为同预算全训。原第4节20轮判定保留为中途健康参考，不作为门。正式配置改为`configs/train/bdd100k-widen-r2-control-640-e200-b32.yaml`（无stage暂停）。本修订发生在任何训练启动之前，不构成事后改门。

## 1. 目的

隔离ACR收益的来源：small AP增益来自（a）P2分支单纯加宽（容量/算力），还是（b）非对称重分配（加宽P2同时减配P5的设计本身）。这是答辩与审稿对"GMACs +17.08%买到增益"质疑的关键防线。

## 2. 冻结设计

- 模型：`configs/models/yolo26n-p2-widen.yaml`——仅L19加宽32→64（与ACR完全一致），L28保持官方256（不减配）。其余逐层与官方相同。
- 成本定位：参数 official < ACR(-6.45%) < widen-only（widen-only比ACR更多参数与算力，单元测试断言排序）。因此若ACR ≥ widen-only，则重分配设计在更少算力下不劣于纯加宽，"非对称"主张成立。
- 训练：`configs/train/bdd100k-widen-r2-control-640-e200-stop20-b32.yaml`——与ACR Stage20协议逐字段一致（BDD 70k/10k、640、200轮计划epoch20暂停、b32/nbs64、seed0、GPU1）。
- 预训练：同ACR，L19/L20/L29随机初始化，其余官方权重加载。

## 3. 对照基准（全部使用冻结值，不重跑）

| 组 | small AP @20ep | 来源 |
|---|---|---|
| R0 官方P2 | 0.083358 | `results/rdci-stage20/summary.json` |
| ACR | 0.091532 | `results/acr-stage20/summary.json` |

## 4. 预注册判定（训练前冻结，200轮口径）

以R2（widen-only）@200ep完整10k尺寸评估为准，对照冻结值R0-200（small AP 0.103266、all AP 0.269294）与ACR-200（small AP 0.110785、all AP 0.280471）：

- **重分配必要性成立**：R2 small AP < ACR-200（0.110785），即纯加宽（更多参数与算力）不能复现ACR增益；P5减配是设计贡献而非代价。
- **宽度解释成立（诚实风险）**：R2 small AP ≥ 0.110785——增益主要由P2宽度解释，ACR叙事降级为"加宽有效且P5减配可补偿参数与速度"，论文如实改写。
- 附加记录：三方（R0/ACR/R2）参数、GMACs、all/small AP/AR完整对照表；如两结果接近（差距<1%相对值），如实记录为"宽度与重分配贡献不可分"。
- 无论哪种结果，不追加变体、不调宽度重跑；本对照是一次性预注册。

## 5. 执行步骤

1. [已由代理完成] 单元测试（4项）；提交前跑全量测试与审计。
2. Stage20：`scripts/run_experiment.py --config configs/train/bdd100k-widen-r2-control-640-e200-stop20-b32.yaml`（约4小时）。
3. 评估：`scripts/evaluate_object_sizes.py --model runs/detect/phase18c-widen-stage20/bdd100k-yolo26n-p2-widen-r2-control-640-e200-stop20-b32-seed0/weights/best.pt --data configs/data/bdd100k.yaml --output results/acr-stage20/r2-widen-size-val-ep20.json --device 1`。
4. 判定与收口写入`results/acr-stage20/widen_control_summary.json`并同步状态文档。

## 6. 禁止事项

- 不因结果调整设计重跑；不把widen-only结果用于任何选型。
- 在线曲线不作判定依据；判定只认完整10k尺寸评估。
