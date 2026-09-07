# 代码、版本与实验产物索引

> 更新日期：2026-09-07。本文描述生命周期，不替代`PROJECT_STATUS.md`中的数值结论。

## 1. Git与工作树

根项目因环境限制使用`.project.git`保存Git元数据；开发版本通过工作树隔离。工作树不是发布版本，也不是大文件归档。

| 工作树/分支 | 阶段 | 状态 | 约占用 |
|---|---|---|---:|
| 根目录 / `master` | 权威主线 | 当前活动；含P8-P18已合并证据、论文与清理记录 | 根项目总量约32GB（服务器） |
| `p2-early-channel-attention` | 通道注意力 | 已关闭，暂留未规范化产物 | 1.6GB |
| `p4-p2p3-distillation` | 跨层蒸馏 | 已关闭，暂留未规范化产物 | 1.3GB |
| `p5-detail-context` | RDCI | 已关闭，暂留未规范化产物 | 693MB |
| `p6-perspective-copy-paste` | PASCP | 已关闭，暂留未规范化产物 | 690MB |
| `p7-perspective-sparse-refinement` | PGSR与路由 | 已完成；保存正式原始预测副本 | 2.3GB |
| `p12-pre-inference-router` | 检测前效用路由 | 已完成并并入`master`；开发门失败 | 动态 |
| `p13-visdrone-resolution-oracle` | VisDrone分辨率上限 | 已完成并并入`master`；预注册上限门通过 | 动态 |
| `p14-scout-resolution-router` | ScoutDet-RV集合路由 | 已完成并并入`master`；内部门失败，路线关闭 | 动态 |
| `p17-uavdt-external-validation` | 外部验证 | 在途分支；证据留在服务器工作树 | — |
| `feat/p18-*`（合入`master`） | ACR/widen/seed1/seed2 | 已全部合入主线并封箱 | — |

P16已移除P8、P9、P11和P15工作树及对应已合并本地分支；这些阶段没有独有实验产物。P9机器记录中的历史绝对路径保持原样，所引用的checkpoint、score和manifest均有SHA256一致的根目录规范副本。

**2026-09-07异地备份恢复说明**：Windows副本（`D:\yolo`）不含`.worktrees/`（未打包，约7.3GB），已执行`git worktree prune`清除失效注册；分支内容全部在git历史中，但p7审计根、p13/p14逐图预测等未跟踪证据仅存于服务器，异地使用需单独打包。P18实验线（ACR/widen/两份seed复制）全部在根工作树完成并提交。

历史工作树只有在以下条件全部满足后才能删除：分支已提交；忽略产物已列清单；需保留的权重、日志和预测已有规范副本与SHA256；对应报告不依赖仅存于该工作树的路径。P2/P4/P5/P6仍有唯一训练日志、预测或路径迁移关系，P7是完整外部证据审计根，P13/P14保留逐图预测和内部筛选证据，因此继续保留。

## 2. 源码生命周期

| 路径 | 内容 | 状态 |
|---|---|---|
| `configs/models/yolo26n-p2-acr.yaml` | **ACR最终方法**：L19 P2检测分支32→64、L28 P5头分支256→192，纯YAML零自定义代码 | P18封箱的正向自研结构改动 |
| `configs/models/yolo26n-p2-widen.yaml` | widen-only容量对齐对照 | P18C已收口；与ACR统计不可分 |
| `sdf/refinement/router.py` | 透视分带路由特征与轻量头 | 正向候选；未通过启发式small AP门 |
| `sdf/refinement/scout_router.py` | 320 scout检测集合价值与不确定性路由头 | P14内部+3.35%未过+5%门；历史候选 |
| `sdf/refinement/sparse.py` | 稀疏裁块/细化辅助 | PGSR裁块路线已关闭 |
| `sdf/modules/channel_attention.py` | P2浅层通道注意力 | 历史失败候选 |
| `sdf/modules/detail_context.py` | RDCI细节-上下文注入 | 历史失败候选 |
| `sdf/modules/dwf.py` | 动态加权融合及降级变体 | 历史失败候选 |
| `sdf/modules/haar_downsample.py` | Haar小波无损下采样 | P18A已关闭；作为ACR容量假设的反证消融保留 |
| `sdf/modules/sdd.py` | 细节保持下采样 | 历史失败候选 |
| `sdf/losses/sa_nwd.py` | 尺度感知NWD-IoU损失 | 历史失败候选 |
| `sdf/augmentations/perspective_copy_paste.py` | PASCP训练增强 | 历史失败候选 |
| `upstream/` | Ultralytics 8.4.104冻结源码及有限接入 | 依赖快照；不做无关升级 |

失败模块保留是为了加载checkpoint、复现实验和支持负结果审计。它们默认关闭，不应再组合成“YOLO26-SDF最终模型”。**ACR不依赖任何`sdf`代码或`upstream`补丁**：官方P2 backbone逐层不变，`upstream/ultralytics/nn/tasks.py`中的自定义注册只服务于已关闭路线。

## 3. 配置生命周期

- `configs/data/`：当前数据入口与数据说明，保留。
- `configs/eval/`：PGSR/路由冻结清单，禁止改写后冒充原协议。
- `configs/models/yolo26n-p2-acr.yaml`、`yolo26n-p2-widen.yaml`、`yolo26n-p2-hwd.yaml`：P18模型结构配置（ACR为最终方法）。
- `configs/train/bdd100k-640.yaml`、`bdd100k-p2-640.yaml`：正式基线配置，保留。
- `configs/train/bdd100k-acr-*.yaml`（smoke/R1/S1/S2）、`bdd100k-widen-r2-control-*.yaml`：P18最终方法训练与复制配置，保留。
- `configs/train/visdrone-*.yaml`：已执行外部数据协议；test-dev相关结果封存。
- 其余带`sdd`、`dwf`、`sa-nwd`、`channel-attention`、`p2p3`、`rdci`、`pascp`、`resolution-960`名称的配置均为历史消融配置，只用于复现，不代表待运行任务。

不移动历史配置，因为报告、manifest和受控启动命令记录了原路径；目录层面的整洁不能以破坏复现链为代价。

## 4. 证据层级

1. `results/**/summary.json`、尺寸指标JSON、CSV、manifest和冻结NPZ/JSONL：机器证据。
2. `results/reports/`：按主题归档的人读报告，由机器证据解释结论。
3. `docs/paper/RESULTS.md`：论文主表，只摘录已审计结果。
4. `PROJECT_STATUS.md`：项目状态，不取代底层证据。

集成运行时机器证据位于`results/integrated-router-runtime/benchmark.json`，包含实现提交、环境、模型/路由/清单哈希、选图集合哈希、三次原始计时和汇总。

P13机器证据位于`results/visdrone-resolution-oracle/`。小型指标、Oracle标签和门控JSON进入Git；约16MB/份的640、960及混合逐图预测由`.gitignore`排除并保留在P13工作树，哈希记录在报告中。

P18机器证据位于`results/acr-stage20/`（ACR/R0/widen三方、200轮终评、四份Bootstrap、三seed终表、推理速度基准）与`results/haar-downsample-stage20/`（P18A负结果）。进入Git的是JSON指标与Bootstrap；约270MB/份的10k逐图预测`.predictions.jsonl`由`.gitignore`排除，其中R0/ACR/widen的200轮终评预测有规范副本保留在根目录（2026-09-06清理仅删除ep20/ep40中间预测，Bootstrap JSON保留，checkpoint可再生）。报告：`results/reports/bdd-modules/acr_stage20_report.md`（§1-9完整证据链）。

大体积原始路由预测的规范副本位于根项目`results/perspective-router-final/`，共六个JSONL，约1.65GB；`p7`工作树保留相同副本及小型机器结果。提交到Git的是哈希、指标、manifest和报告，不提交大JSONL。

P12的train/dev缩略图NPZ是由冻结图像清单和脚本确定性生成的派生缓存，P16已删除；JSON元数据、seed0权重、指标和停止门仍保留。需要复查时应以冻结协议重建，不得继续调参或访问正式val。

## 5. 清理规则

- 可直接删除：缓存、可重建临时输出、内容已被当前文档完全替代的错误计划。
- 归档不执行：失败路线协议、配置、源码和报告。
- 不可直接删除：checkpoint、正式预测、manifest输入、唯一训练日志、test-dev唯一结果。
- 每次清理后必须运行`scripts/audit_repository_layout.py`、`scripts/audit_paper_claims.py`、完整单元测试和`git diff --check`。

最近一次清理记录：`results/reports/analysis/p18_repository_cleanup_report.md`（2026-09-06，ACR中间预测清理）；更早为`p16_repository_cleanup_report.md`。
