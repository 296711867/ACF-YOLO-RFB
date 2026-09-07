# YOLO26-SDF 项目状态

> 本文件是项目当前状态的唯一入口（single source of truth）。  
> 最后更新：2026-09-07，Asia/Shanghai
> 当前阶段：**P18实验线封箱（ACR三seed全过门）——论文写作期 + 仓库开源整理**
> 新会话交接：`docs/internal/SESSION_HANDOFF.md`

## 1. 当前结论

YOLO26n 640正式基线和官方P2-640独立消融均已验收；当前BDD有效主模型仍为官方P2-640。8,769参数透视分带路由在正式10k val上相对全640取得all AP +4.31%、small AP +13.75%、small AR +8.76%，第一次形成自研模块相对主模型的完整数据正收益。1,000次配对Bootstrap中，all/small AP差值95%区间分别为[+0.00942,+0.01338]和[+0.01102,+0.01693]。但其small AP点估计0.115461略低于等预算粗检启发式0.116185，违反预注册严格门；两者all/small AP差值区间均跨0，不能证明学习头更优。集成运行时中学习25%为3.7802±0.0114秒，比全960的3.6136±0.0868秒慢4.61%，也不具备效率优势。因此它仍是正向精度候选而非已证明必要、实用的最终原创方法。VisDrone test-dev继续永久封存。

P11已完成中英文摘要、研究问题、相关工作、方法、实验、结果、讨论和结论组成的无编号整合稿，并建立学校模板接入清单与正文回归审计。正式章节编号、图表编号、个人信息和参考文献样式仍等待学校模板与论文元数据。

P12按冻结协议实现23,417参数检测前缩略图路由、连续效用排序损失和同成本统计基线。seed0在500图开发集的utility capture为0.36375，低于固定岭回归图像统计基线0.41027，相对低11.34%。由于协议要求三个seed全部超过最佳图像基线，seed0已触发停止门；未运行seed1/2、BDD正式val、检测运行时或外部数据，路线关闭。

P13在推理前冻结VisDrone原生YOLO26n seed0模型、548张val、640/960和25% GT Oracle。全960相对全640的small AP/AR为+42.72%/+29.35%，25% Oracle达到all/small AP/small AR `0.201256/0.125729/0.261011`，保留全960的64.77% small AP绝对增量，七项继续门全部通过。该结果只证明跨数据集分辨率价值上限；VisDrone val参与过历史模型选择，不是最终未见外部验证，test-dev未再次访问。

P14已冻结并执行ScoutDet-RV：用320 scout检测集合预测960相对640的边际价值，不读取RGB缩略图；按208个视频序列分组避免相邻帧泄漏，三seed以均值减总标准差执行25%预算路由。train内部holdout上三seedutility capture为0.83602/0.82904/0.83493，均超过最强HistGradientBoosting基线0.80975；集成为0.83690，但相对基线仅+3.35%，低于预注册+5%门。P14因此在val前关闭，没有访问VisDrone val/test-dev，也没有运行Bootstrap或端到端计时。

P15已把P13/P14完整整合到中英文摘要、研究问题、相关工作、方法、结果、讨论、局限和结论，并建立SCI投稿定位。当前中心论点为：BDD与VisDrone均存在高分辨率价值，但BDD学习路由未超过简单启发式，VisDrone ScoutDet-RV也未达到相对强树基线的预注册幅度；论文贡献是预算受控、跨数据、强基线和开放负结果的实证证据，而非新的SOTA检测器或已验证动态路由算法。

P16完成仓库生命周期整理：删除245个Python/ruff/pytest缓存目录、P12的两个可重建缩略图NPZ缓存，以及已合并且无独有实验产物的P8/P9/P11/P15工作树和本地分支。项目占用由34,518,523,904降至34,055,340,032字节，释放463,183,872字节（441.7MiB）。P9三个运行时输入已与根目录规范副本逐项核对SHA256一致；P2/P4/P5/P6、P7、P13和P14仍含唯一训练日志、正式预测、外部审计或失败路线证据，继续保留。

P18A（2026-08-27/28，用户授权的筛选级证据）：Haar小波无损下采样Stage20筛选已按预注册协议执行并关闭。将官方P2最浅两处stride-2卷积替换为固定正交DWT加3×3融合（参数+15,120即+0.57%，GMACs +10.23%），20轮在线终点反超控制组（0.2344 vs 0.2338），但完整10k尺寸评估small AP 0.082618，相对冻结R0的0.083358为-0.89%，低于关闭线+1.0%；medium +0.89%、small AR +0.58%、large -1.12%。无损下采样假设在BDD-640不成立，路线关闭，不追加变体。协议与报告：`docs/protocols/HAAR_DOWNSAMPLE_STAGE20_PROTOCOL.md`、`results/reports/bdd-modules/haar_downsample_stage20_report.md`。

P18B（2026-08-28/29，用户授权的筛选级证据）：非对称容量重分配（ACR）Stage20筛选**通过预注册继续门**。L19 P2检测分支32→64通道、L28 P5头分支256→192通道，backbone不变（预训练权重除L19/L20/L28/L29外全部加载），参数净减6.45%、GMACs +17.08%。完整10k尺寸评估：all AP 0.246022（+4.61%）、small AP 0.091532（**+9.81%**，超+3.0%门三倍余量）、small AR 0.278957（+8.35%）、medium +4.62%、large +0.53%（P5减配近零代价）。这是项目首个通过预注册筛选门的自研结构改动；此前最好为蒸馏D2的+1.43%。冷启动极小（epoch1即-1.1%、epoch2起领先）。协议与报告：`docs/protocols/ACR_STAGE20_PROTOCOL.md`、`results/reports/bdd-modules/acr_stage20_report.md`；机器证据`results/acr-stage20/`。

P18B epoch40配对确认（2026-08-30）：ACR与R0均从epoch20精确恢复至40（R0经CUDA_VISIBLE_DEVICES=1与ACR同物理GPU配对，偏差已记录）。small AP增益保持：**+9.16%**（0.095113 vs 0.087132），未现通道注意力式长训练掉队；all +3.54%、small AR +7.84%、large -1.21%（P5减配代价显现但仍远小于small收益）。在线0.2505 vs 0.2419。决策进入200轮配对全训。证据：`results/acr-stage20/stage40_confirmation.json`及`bootstrap-acr-vs-r0-ep40.json`。

P18C 容量对齐对照（2026-09-05）：widen-only（L19加宽同ACR、L28不减配）200轮全训完成。三方同协议消融：R0/ACR/widen的small AP为0.103266/0.110785/0.109599（相对R0分别+0%/+7.28%/+6.12%），ACR对widen配对Bootstrap差值CI均跨0。**判定：宽度与重分配贡献不可分**——P2加宽是small增益主导因素；非对称重分配以三者最低参数（2,490,624，较widen -7.9%）取得同等精度，定位为效率型贡献。协议`docs/protocols/WIDEN_CONTROL_STAGE20_PROTOCOL.md`，证据`results/acr-stage20/widen_control_summary.json`，报告§9。

P18三seed终局（2026-09-07）：seed0/1/2全部通过预注册跨seed门，small AP相对R0-200分别为+7.28%/+5.93%/+5.24%；三seed均值0.109619±0.001070（CV 0.98%），all AP 0.279791±0.000712，small AR 0.309059±0.003473，无剔除。推理速度基准：ACR 5.844ms/图 vs 官方P2 6.103ms/图（-4.24%，FP32 b16独占GPU0）。**P18实验线正式封箱**；剩余工作为论文写作与导师追认创新口径。证据：`results/acr-stage20/three_seed_final_summary.json`、`inference_benchmark_gpu0.json`；论文结果主表已收录（`docs/paper/RESULTS.md`§1.10，含版本状态说明）。

P18B 200轮全训闭环（2026-09-01）：**ACR路线以完整三级证据链闭环为正向自研结构改动**。(1) 同协议配对终评（R0-200亦完成）：all AP 0.280471 vs 0.269294（+4.15%）、small AP 0.110785 vs 0.103266（**+7.28%**）、small AR +6.99%、large +1.35%、medium -1.98%（如实记录的代价）；配对Bootstrap all/small差值CI `[+0.00848,+0.01338]`/`[+0.00470,+0.01062]`均严格>0。(2) 对官方b64主模型：all +4.57%、small +9.17%，Bootstrap CI亦严格>0。(3) 增益轨迹+9.81%(20轮)→+9.16%(40轮)→+7.28%(200轮)稳定。全部以-6.45%参数（2,490,624 vs 2,662,400）达成，GMACs +17.08%。待办：导师追认创新口径；容量对齐对照（widen-only）；多seed决策。完整报告：`results/reports/bdd-modules/acr_stage20_report.md` §7-8；机器证据：`results/acr-stage20/`（含full200_summary.json、三份Bootstrap）。

当前正确顺序是：

```text
Phase 0 冒烟验收（已完成）
  → Git/评估脚本/统计/输出规则冻结（已完成）
  → YOLO26n 200 epoch 正式基线（已完成）
  → 小目标专项评估（已完成）
  → 官方 P2 消融（已完成并验收）
  → SDD 工程 smoke（已完成）
  → SDD 配对短跑（已完成，淘汰）
  → DWF 工程 smoke（已完成）
  → 完整 DWF 配对短跑（已完成，失败）
  → DWF Variant A nearest+gate 工程验收（已完成）
  → DWF Variant A 配对筛选（已完成，整体门槛失败）
  → DWF Variant B 工程验收（已完成）
  → DWF Variant B 配对筛选（已完成，失败）
  → DWF 路线淘汰、恢复官方 P2（已完成）
  → SA-NWD 工程与 smoke（已完成）
  → P2 control / SA-NWD 配对 50 epoch（已完成，整体门槛未过）
  → SA-NWD small-only 降级诊断（已完成，失败）
  → SA-NWD τ=512 快速衰减（已完成，失败）
  → SA-NWD 路线淘汰、恢复官方 P2（已完成）
  → 官方 P2-960 smoke（已完成）
  → 官方 P2 640/960 配对 50 epoch（已完成，通过）
  → 官方 P2-960 完整 BDD100K 训练（39/200，人工终止，不恢复）
  → epoch 39 完整 10k 独立/尺寸评估（已完成，960工作包关闭）
  → P2-640 融合前轻量通道注意力工程/smoke（已完成，通过）
  → P2-640 融合前轻量通道注意力配对50 epoch（已完成，通过）
  → 完整BDD batch64预检（已完成，通过）
  → 首轮完整BDD 20 epoch诊断（已完成，协议不公平，不作为正式对照）
  → 相同200-epoch计划、epoch20可恢复暂停的公平Stage 20（已完成，通过）
  → 从完整状态精确恢复至epoch40并再次暂停（已完成，通过）
  → 从完整状态精确恢复至epoch60并再次暂停（已完成，通过）
  → 取消后续20轮分段，从epoch61直跑至epoch200（已完成）
  → 完整10k独立验证与尺寸评估（已完成，未通过保留门）
  → 通道注意力路线关闭、恢复官方P2-640（已完成）
  → VisDrone正式协议冻结（已完成）
  → 官方DET格式/算法兼容评估链（已完成）
  → 原生/P2 batch32显存预检（P2 OOM，失败）
  → 两组统一batch16重新预检（已完成，通过）
  → 原生/P2同调度Stage 20与双口径评估（已完成，P2淘汰）
  → 冻结类别频率权重单变量（已完成）
  → 工程与1 epoch预检（已完成，通过）
  → 类别权重Stage 20与双口径评估（已完成，失败）
  → 原生control从epoch20续训至100（已完成）
  → val最终评估与test-dev唯一一次双口径评估（已完成）
  → RESULTS主表与冻结配置多seed方案（已完成）
  → seed1/seed2顺序训练与val评估（已完成）
  → 三seed统计与RESULTS主表更新（已完成）
  → 论文图表、误差分析与结果章节初稿（已完成）
  → 图注、方法衔接、引用库与数值审计（已完成）
  → BDD同图像配对Bootstrap置信区间（已完成）
  → BDD定性漏检/误检和小目标案例分析（已完成）
  → P2到P3小目标蒸馏方案与原创边界冻结（已完成）
  → 异构层映射、损失、checkpoint剥离与3 epoch smoke（已完成，通过）
  → 完整BDD D0-D3公平Stage20与10k尺寸评估（已完成，三候选均未通过）
  → P2到P3蒸馏路线关闭，不进入Stage40/200轮（已完成）
  → 有效自研算法硬要求确认、DCAS官方论文/代码审计（已完成）
  → RDCI原创边界、模块实现与CPU/CUDA回归（已完成）
  → RDCI 3 epoch smoke与batch64显存预检（已完成）
  → 官方P2/RDCI同协议batch32 Stage20与10k尺寸评估（已完成，未通过）
  → RDCI路线关闭，不进入Stage40/200轮（已完成）
  → PASCP文献边界、算法、代码与Stage20协议冻结（已完成）
  → PASCP全量回归、真实数据抽查与3 epoch工程smoke（已完成）
  → 官方P2/PASCP同协议Stage20与10k尺寸评估（已完成，未通过）
  → PASCP路线关闭，不进入Stage40/200轮（已完成）
  → PGSR 500图B0/B1/SAHI/GT Oracle可行性门（已完成；裁块路线关闭）
  → 25%/50%自适应全图分辨率路由GT上限（已完成；25%通过）
  → 互斥5k路由训练集、P2透视分带路由头与固定500图开发门（已完成；三seed通过）
  → seed0冻结路由一次性完整10k val与效率评估（已完成）
  → 正式基础门通过、启发式small AP比较门失败；不在val上重调（已完成，路线关闭）
  → 路由配对Bootstrap与集成运行时补证（已完成；无启发式优越性或速度优势）
  → P11无编号论文整合稿、模板接入清单与声明审计（已完成）
  → P12检测前效用排序路由工程与BDD train/dev筛选（已完成；seed0开发门失败并关闭）
  → P13 VisDrone冻结模型640/960与25% GT Oracle上限（已完成；继续门通过）
  → P14 ScoutDet-RV协议、序列隔离划分、核心模型与CUDA工程门（已完成）
  → VisDrone train冻结320/640/960预测、强基线与三seed内部门（已完成；+3.35%低于+5%，关闭）
  → P15 P13/P14论文整合与SCI实证投稿定位（已完成）
  → 并入正式论文模板并统一章节编号（等待学校模板与个人元数据）
```

## 2. 已完成

### 2.1 工程与环境

- [x] Ultralytics YOLO26 源码冻结到 `upstream/`。
- [x] Ultralytics 版本为 8.4.104。
- [x] `yolo26n.pt` 已保存并记录 SHA256。
- [x] Conda 环境 `yolo26-sdf` 已创建。
- [x] Python 3.11.15、PyTorch 2.6.0+cu126、Torchvision 0.21.0+cu126 已记录。
- [x] 两张 RTX 3090（各 24 GB）可被 PyTorch 使用。
- [x] YOLO26n 权重加载和单张 GPU 预测通过。
- [x] YOLO26n 原生 P3/P4/P5 网络构建通过。
- [x] 官方 YOLO26-P2 P2/P3/P4/P5 网络构建通过。
- [x] `pip check` 无依赖冲突。

### 2.2 接口审计

- [x] 确认 YOLO26 原生 `end2end: true`，属于端到端/NMS-free 路径。
- [x] 确认 `reg_max: 1`。
- [x] 确认损失入口为 `E2ELoss → v8DetectionLoss → BboxLoss`。
- [x] 确认标签分配器为 `TaskAlignedAssigner`。
- [x] 确认官方已经提供 `yolo26-p2.yaml`，P2 不重复实现。
- [x] 明确 SA-NWD 后续必须同时兼容 one-to-many 和 one-to-one 损失分支。

### 2.3 BDD100K

- [x] 100K 图像已下载：70,000 train、10,000 val、20,000 test。
- [x] 真实 train/val 检测标注已转为 80,000 份 YOLO 标签。
- [x] 排除 500 条第三方镜像附带的 `synthetic_val_*` 数据。
- [x] 跳过 447 个零面积/退化框，移除 1 个完全重复框。
- [x] 图像与标注跨镜像抽样 SHA256：10/10 一致。
- [x] Ultralytics 配置入口全量扫描：80,000 images、80,000 labels、0 corrupt。
- [x] 物理目录已规范化为 `images/train|val|test` 和 `labels/train|val`。

### 2.4 VisDrone2019-DET

- [x] train、val、test-dev 已全部下载。
- [x] 6,471 train、548 val、1,610 test-dev 已转换为 YOLO 格式。
- [x] 移除 4 个完全重复框。
- [x] Ultralytics 全量扫描：8,629 images、8,629 labels、0 corrupt。

### 2.5 Phase 0 冒烟验收

- [x] 人工检查 200 张 BDD100K 和 50 张 VisDrone 标签可视化，未见系统性偏移或类别错位。
- [x] 冻结 BDD100K 400 train / 100 val 子集并记录清单 SHA256。
- [x] 子集包含全部 10 类；train 7,251 实例，val 1,931 实例，空标签均为 0。
- [x] 原生 YOLO26n 完成 3 epoch train、独立 val 和固定 10 图 predict。
- [x] 39 个训练/验证 loss 与指标值全部有限，无 NaN/Inf。
- [x] 优化器实际解析为 AdamW，lr 0.000714，momentum 0.9。
- [x] args、CSV、best/last 权重、曲线、验证图和预测图完整保存。
- [x] 验收报告：`results/reports/baselines/phase0_smoke_report.md`。

### 2.6 Phase 1 正式基线准备

- [x] 实现 COCO 风格 small/medium/large AP/Recall 评估，并通过合成与真实 smoke 自测。
- [x] BDD100K 80,000 张、1,473,535 个有效实例的类别/尺寸/场景统计落盘。
- [x] 全量统计严格复现 447 个退化框和 1 个重复框过滤结果。
- [x] 冻结 640/960 正式配置、batch、seed、run 名称和输出目录；实际设备按资源审计记录。
- [x] 受控启动器记录 commit、环境和文件哈希，拒绝脏树与覆盖已有 run。
- [x] 使用 `.project.git` 解决根目录只读空 `.git` 挂载限制。
- [x] 验收报告：`results/reports/baselines/p1_baseline_prep_report.md`。

### 2.7 YOLO26n 640 正式基线

- [x] 完成 BDD100K 200/200 epoch，无 early stopping、NaN 或 Inf。
- [x] 最佳 epoch 181：P 0.68630、R 0.42203、mAP50 0.46424、mAP50-95 0.25473。
- [x] 完整 10k val 独立验证通过，与训练结束验证一致。
- [x] 尺寸 AP50-95：small 0.08877、medium 0.29092、large 0.50821。
- [x] best/last、CSV、曲线、元数据、resume history 和原始预测证据完整。
- [x] 验收报告：`results/reports/baselines/phase1_baseline_640_report.md`。

### 2.8 官方 YOLO26-P2 准备

- [x] 直接复用官方 P2 YAML，无自定义模块。
- [x] stride `[4, 8, 16, 32]`，BDD 10 类参数 2,518,352、7.5 GFLOPs。
- [x] 同一 `yolo26n.pt` 预训练权重迁移 360/902 项，新增层随机初始化。
- [x] 400/100 固定子集、batch 64、1 epoch smoke 通过，峰值显存 20.8 GB。
- [x] 正式配置 `configs/train/bdd100k-p2-640.yaml` 已冻结。
- [x] 准备报告：`results/reports/baselines/p1_official_p2_prep_report.md`。

### 2.9 官方 YOLO26-P2 独立消融

- [x] 完成 BDD100K 200/200 epoch，无 early stopping、NaN、Inf 或 OOM。
- [x] 最佳 epoch 183：P 0.67420、R 0.43414、mAP50 0.47706、mAP50-95 0.26871。
- [x] 完整 10k val 独立验证通过：P 0.674、R 0.434、mAP50 0.477、mAP50-95 0.269。
- [x] 尺寸 AP50-95：small 0.101480、medium 0.321219、large 0.519476。
- [x] 相对基线：整体 AP50-95 +5.01%，small +14.32%，medium +10.42%，large +2.22%。
- [x] 参数仅增加 0.42%，但 GFLOPs 增加 29.3%，训练时间约 2.05 倍，峰值显存接近 24 GB。
- [x] 验收报告：`results/reports/baselines/phase1_official_p2_640_report.md`；P2 保留进入后续工作包。

### 2.10 SDD 最小实现与固定 smoke

- [x] 实现 `PixelUnshuffle(2) → 1×1 → DW 3×3 → 1×1 → BN → SiLU`，仅替换生成 P2 的下采样。
- [x] 模块位于 `sdf/`，上游仅增加 YAML 注册点；通过 editable 安装支持控制台和项目外加载。
- [x] 空间重排可逆、shape、FP32/FP16 梯度、序列化、YAML、stride 与 TorchScript 测试通过。
- [x] BDD100K 400/100、640、batch 64、3 epoch smoke 完成，最高日志显存约 22.3 GB。
- [x] 全部 CSV 数值有限，train/val/save/final val 与项目外 checkpoint 重载通过。
- [x] BDD 10 类参数 2,517,104，较官方 P2 少 1,248；GFLOPs 均约 7.5。
- [x] 报告：`results/reports/bdd-modules/p2_sdd_prep_report.md`。

### 2.11 SDD 配对 50 epoch 筛选

- [x] P2 control 与 P2+SDD 除 model/name 外配置完全一致，共用原始 `yolo26n.pt`、数据、seed 和 GPU0。
- [x] 两组均完成 50/50 epoch，无 early stopping、NaN、Inf 或 OOM，元数据状态为 `completed`。
- [x] 最佳 mAP50-95：control 0.05975，SDD 0.02918，下降 51.16%。
- [x] 尺寸 AP50-95 相对变化：all -49.78%、small -61.48%、medium -60.27%、large -46.74%。
- [x] SDD 在 epoch 10/20/30/40/50 均落后，满足淘汰条件；不启动完整 200 epoch。
- [x] SDD 代码和产物仅为复查负结果保留，不进入后续活动网络。
- [x] 报告：`results/reports/bdd-modules/p2_sdd_screen_report.md`。

### 2.12 DWF 最小实现与固定 smoke

- [x] 仅替换 P3→P2 融合，包含零初始化动态采样、全局权重、局部门控和轻量融合；不含 SDD/SA-NWD。
- [x] bilinear 初始对齐、权重归一化、shape、FP32/FP16 梯度、序列化、YAML、stride 和 TorchScript 通过。
- [x] BDD100K 400/100、640、batch 32、3 epoch smoke 完成，最高日志显存约 14.6 GB。
- [x] 全部 CSV 数值有限，train/val/save/final val 与项目外 checkpoint 重载通过。
- [x] BDD 10 类参数 2,530,203、8.0 GFLOPs，较官方 P2 分别增加约 0.47% 和 6.7%。
- [x] 记录 CUDA grid-sample 反向非位级确定性的限制，后续使用配对与多 seed 控制。
- [x] 报告：`results/reports/bdd-modules/p2_dwf_prep_report.md`。

### 2.13 完整 DWF 配对 50 epoch 筛选

- [x] P2 control 与 P2+DWF 除 model/name 外配置一致，共用原始权重、数据、batch 32、seed 和 GPU0。
- [x] 两组完成 50/50 epoch，无 early stopping、NaN、Inf 或 OOM；DWF 非位级确定性警告已记录。
- [x] 最佳 mAP50-95：control 0.06737，DWF 0.05820，下降 13.61%。
- [x] 尺寸 AP50-95：all -19.78%、small -2.41%、medium -11.47%、large -23.73%。
- [x] DWF 未提升 small AP/AR 且整体退化，不启动完整 200 epoch。
- [x] 进入方案第 11.2 节的 nearest/无门控降级诊断，暂不最终删除 DWF。
- [x] 报告：`results/reports/bdd-modules/p2_dwf_screen_report.md`。

### 2.14 DWF Variant A 工程验收

- [x] 仅将完整 DWF 的 DySample 替换为 nearest 2×，保留全局权重、局部门控和轻量融合。
- [x] 模型中恰好 1 个 `DWFNearestGate`、0 个 DySample、0 个 SDD，stride 为 `[4, 8, 16, 32]`。
- [x] CPU 单元、实际 RTX 3090 CUDA/AMP 反向、梯度和 checkpoint 重载通过。
- [x] 固定 400/100、640、batch 32、3 epoch smoke 完成，最高日志显存约 14.7 GB。
- [x] 无 early stopping、OOM、NaN 或 Inf；best/last、最终验证和元数据完整。
- [x] 报告：`results/reports/bdd-modules/p2_dwf_nearest_gate_prep_report.md`。

### 2.15 DWF Variant A 配对 50 epoch 筛选

- [x] 完成 50/50 epoch，无 early stopping、OOM、NaN 或 Inf；best 独立回验一致。
- [x] Ultralytics mAP50-95 为 0.05809，较 batch 32 control 下降 13.78%。
- [x] 尺寸 AP50-95：small +14.87%，all -15.01%，medium -7.11%，large -13.45%。
- [x] small AP 有恢复但 small AR 下降 8.10%，且整体退化明显，不进入 200 epoch。
- [x] 报告：`results/reports/bdd-modules/p2_dwf_nearest_gate_screen_report.md`；下一步去除局部门控。

### 2.16 DWF Variant B 工程验收

- [x] 使用 nearest+归一化全局权重，移除 DySample 和局部门控；不含 SDD/SA-NWD。
- [x] CPU 单元、实际 RTX 3090 CUDA/AMP、YAML、梯度和 checkpoint 重载通过。
- [x] stride `[4, 8, 16, 32]`；BDD 10 类 2,529,554 parameters、约 8.0 GFLOPs。
- [x] 固定 400/100、640、batch 32、3 epoch smoke 完成，最高日志显存约 14.5 GB。
- [x] 无 early stopping、OOM、NaN 或 Inf；best/last、最终验证和元数据完整。
- [x] 报告：`results/reports/bdd-modules/p2_dwf_nearest_weight_prep_report.md`。

### 2.17 DWF Variant B 配对筛选与路线终止

- [x] 完成 50/50 epoch，无 early stopping、OOM、NaN 或 Inf；best 独立回验及尺寸评估完成。
- [x] 尺寸 AP50-95 相对 control：all -17.41%、small -12.40%、medium -3.75%、large -17.15%。
- [x] 完整 DWF、nearest+gate、nearest+weight-only 均未通过整体门槛。
- [x] DWF 路线正式淘汰，不做 200 epoch、不与 SA-NWD 叠加；活动网络恢复官方 P2 concat。
- [x] 报告：`results/reports/bdd-modules/p2_dwf_nearest_weight_screen_report.md`。

### 2.18 SA-NWD 工程与固定 smoke

- [x] 实现 640 参考尺度的尺寸感知 NWD-CIoU 混合；C=32、τ=1024、alpha 上限 0.8。
- [x] 默认关闭路径与原生 CIoU 逐位一致；原生 reg_max=1 L1、分类和 TaskAlignedAssigner 不变。
- [x] one-to-many/one-to-one 双分支均接入相同 SA-NWD；top-k 和动态损失权重不变。
- [x] 15 项回归测试及 RTX 3090 CUDA/AMP 通过；640/960 参考尺度一致性通过。
- [x] 固定 400/100、640、batch 32、3 epoch smoke 完成，最高日志显存约 12.5 GB。
- [x] loss/CSV 全部有限；第 3 epoch 发生 1 次内置 fitness 自动恢复并成功完成，列为筛选监控项。
- [x] 报告：`results/reports/bdd-modules/p2_sa_nwd_prep_report.md`。

### 2.19 SA-NWD 首轮配对 50 epoch 筛选

- [x] control/candidate 均完成 50/50 epoch，无 early stopping、OOM、NaN、Inf 或自动恢复。
- [x] 尺寸 AP：small +6.90%、all -5.28%、medium -7.15%、large +3.26%。
- [x] 尺寸 AR：small +1.60%，但整体 AP 退化，不进入 200 epoch。
- [x] 报告：`results/reports/bdd-modules/p2_sa_nwd_screen_report.md`；下一步仅对 small 区间启用 NWD。

### 2.20 SA-NWD small-only 降级筛选

- [x] 完成 50/50 epoch，无自动恢复、NaN、Inf、OOM 或 early stopping。
- [x] 尺寸 AP：small +3.43%、all -6.75%、medium -8.91%、large -7.42%；small AR -2.02%。
- [x] 未通过正式门槛；不进入 200 epoch，转入 τ=512 最终快速衰减诊断。

### 2.21 SA-NWD τ=512 最终诊断与路线终止

- [x] 完成 50/50 epoch，无自动恢复、NaN、Inf、OOM 或 early stopping。
- [x] 尺寸 AP：small +2.37%、all -6.52%、medium -7.79%、large +0.51%；small AR -0.36%。
- [x] 连续、small-only、快速衰减三组均未通过正式门槛，SA-NWD 正式淘汰。
- [x] 不做 200 epoch、不继续参数搜索或模块叠加；代码默认关闭，仅保留负结果复查。
- [x] 总报告：`results/reports/bdd-modules/p2_sa_nwd_screen_report.md`。

### 2.22 官方 P2-960 工程 smoke

- [x] 冻结 640/960 两组 batch 16、nbs 64 配置，差异仅为 imgsz 和 run name。
- [x] P2-960 完成 3/3 epoch，无 early stopping、自动恢复、OOM、NaN 或 Inf。
- [x] 峰值训练日志显存约 14.7 GB；best/last、最终验证、重载和 stride 检查通过。
- [x] 报告：`results/reports/bdd-modules/p2_resolution_prep_report.md`。

### 2.23 官方 P2 640/960 配对筛选

- [x] 两组完成 50/50 epoch，无 early stopping、自动恢复、OOM、NaN 或 Inf。
- [x] 960 相对 640：all AP +40.49%、small AP +102.58%、small AR +56.61%。
- [x] medium/large AP 分别 +46.32%/+7.77%，所有尺寸 AP/AR 均提升。
- [x] 成本：训练时间约 +105.98%，GFLOPs +125%，参数量不变。
- [x] P2-960 通过候选门槛，报告：`results/reports/bdd-modules/p2_resolution_screen_report.md`。

### 2.24 P2-960 完整训练终止决策

- [x] 完成39/200 epoch，best mAP50-95 0.28507；best/last、optimizer、scaler、EMA完整。
- [x] 人工Ctrl-C安全停止，GPU0释放；元数据明确记录`KeyboardInterrupt`而非训练故障。
- [x] 用户决定不再恢复200 epoch；P2-640保持论文主模型。
- [x] 后续只做epoch39完整10k评估，报告：`results/reports/bdd-modules/p2_resolution_960_pause_report.md`。

### 2.25 P2-960 epoch39完整评估与关闭

- [x] 完整10k独立val：P 0.696、R 0.445、mAP50 0.498、mAP50-95 0.285。
- [x] 相对P2-640 e200：all AP +6.11%、small AP +19.46%、small AR +9.83%、large AP -1.23%。
- [x] 保存尺寸JSON和257 MB逐图预测；明确39/200与200/200不是同预算公平对照。
- [x] P2-960工作包关闭且不恢复，报告：`results/reports/bdd-modules/p2_resolution_960_epoch39_report.md`。

### 2.26 P2浅层近恒等通道注意力准备

- [x] 在官方P2 concat的同一层位只对浅层P2分支做通道门控，保留nearest、concat、C3、层编号和原生损失。
- [x] 初始门控严格等于1；只增加4,160参数（约0.16%），stride保持`[4, 8, 16, 32]`。
- [x] shape、参数、梯度、CUDA FP16、序列化、YAML、forward和best checkpoint重载通过。
- [x] 原始`yolo26n.pt`迁移360/904项，与官方P2的360项相同。
- [x] 固定400/100、640、batch32、seed0完成3 epoch；峰值13.6 GB，无NaN、OOM或自动恢复。
- [x] 工程门通过，允许进入配对50 epoch；报告：`results/reports/bdd-modules/p2_channel_attention_prep_report.md`。

### 2.27 P2浅层近恒等通道注意力配对筛选

- [x] 复用同协议官方P2 control；候选配置除model/name外完全一致，均为400/100、640、50 epoch、batch32、seed0。
- [x] 候选完成50/50 epoch，最佳epoch均为40；无early stopping、自动恢复、OOM、NaN或Inf。
- [x] 训练best mAP50-95从0.06737升至0.07459（+10.72%），50 epoch时间仅+0.34%。
- [x] 尺寸AP：all +6.95%、small +9.64%、medium +9.88%、large +11.67%。
- [x] 尺寸AR：all +4.77%、small +1.60%、medium +1.31%、large +18.93%。
- [x] 全部保留门通过；允许进入完整BDD 20 epoch阶段门，不允许直接200 epoch。
- [x] 报告：`results/reports/bdd-modules/p2_channel_attention_screen_report.md`。

### 2.28 P2浅层近恒等通道注意力Stage 20准备

- [x] batch64固定子集1 epoch完成，峰值训练日志显存20.8 GB，无OOM、NaN、Inf或自动恢复。
- [x] 正式协议恢复官方P2完整训练的70k/10k、640、batch64、seed0和原始`yolo26n.pt`初始化。
- [x] 配置只设置20 epoch，不自动进入40/60/200 epoch。
- [x] 正式启动器兼容主工作区`.project.git`与标准Git worktree，继续强制干净树和证据哈希。
- [x] 准备报告：`results/reports/bdd-modules/p2_channel_attention_stage20_prep_report.md`。

### 2.29 首轮完整BDD Stage 20诊断

- [x] 完成20/20 epoch，3.296小时，平均9.89分钟/epoch；无OOM、NaN、Inf、early stopping或自动恢复。
- [x] 最佳epoch 20：P 0.64168、R 0.39521、mAP50 0.43073、mAP50-95 0.23881。
- [x] epoch10→20 mAP50-95增加0.01719，仍上升但末5轮单轮增益缩小至0.00124→0.00048。
- [x] 与官方P2 epoch20的0.23685仅高0.83%，但学习率0.001785 vs 0.0271785，禁止宣称公平提升。
- [x] 正常结束后的last已剥离optimizer/scaler，不能精确resume；该run降级为工程/预算诊断。
- [x] 报告：`results/reports/bdd-modules/p2_channel_attention_stage20_report.md`。

### 2.30 公平Stage 20

- [x] 与官方P2使用相同200-epoch调度，在epoch20保存后由启动器强制暂停。
- [x] 20行CSV全部有限；last为epoch19且optimizer/scaler/EMA完整，planned epochs仍为200。
- [x] 候选epoch20 mAP50-95 0.23576，官方同epoch0.23685，相对-0.46%；mAP50相对-0.32%。
- [x] epoch10→20增加0.01865，末5轮均为正增益；相对差距从epoch10的-1.17%缩小。
- [x] 完整10k独立val与尺寸画像完成：small AP 0.085917、small AR 0.254950。
- [x] 官方未保存epoch20权重，尺寸结果只作候选绝对画像，不冒充公平差分。
- [x] Stage 20通过，允许从last精确恢复到epoch40；报告：`results/reports/bdd-modules/p2_channel_attention_fair_stage20_report.md`。

### 2.31 公平Stage 40

- [x] epoch21前识别并安全中止一次错误恢复；CSV与Stage20 checkpoint未受污染。
- [x] Trainer在resume参数所有权处禁用pretrained覆盖，正式恢复确认904/904项迁移、904张量精确匹配EMA。
- [x] epoch40按设计暂停；40行CSV全部有限，last含optimizer/scaler/EMA且`epoch=39`、planned epochs=200。
- [x] 候选mAP50-95 0.24342，官方同epoch0.24584，相对-0.98%；epoch20→40增加0.00766。
- [x] 完整10k尺寸画像：small AP 0.088533、AR 0.261950，较Stage20分别+3.05%/+2.75%；各尺寸均改善。
- [x] Stage 40通过，允许从last精确恢复到epoch60；报告：`results/reports/bdd-modules/p2_channel_attention_fair_stage40_report.md`。

### 2.32 公平Stage 60与直跑决策

- [x] epoch60按设计暂停；60行CSV全部有限，last含optimizer/scaler/EMA且`epoch=59`、planned epochs=200。
- [x] 候选mAP50-95 0.25013，官方同epoch0.25100，相对-0.35%；Recall 0.40283 vs 0.40275。
- [x] epoch40→60增加0.00671，最近10轮净增0.00225；累计成本较官方同epoch约+0.86%。
- [x] 用户决定取消后续每20轮暂停；允许从epoch61直接续至200，保留`patience=50`正常早停。
- [x] 决策报告：`results/reports/bdd-modules/p2_channel_attention_fair_stage60_decision.md`。

### 2.33 通道注意力200轮最终评估与关闭

- [x] 2026-08-03 09:20完成200/200；无early stopping、OOM、NaN、Inf或自动恢复，最佳epoch190 mAP50-95 0.26904。
- [x] 完整10k独立val复现P/R/mAP50/mAP50-95约0.689/0.430/0.479/0.269。
- [x] 同预算尺寸AP：all +0.12%、small +0.14%、medium -4.95%、large -0.41%。
- [x] 同预算尺寸AR：all -1.00%、small -1.70%、medium -2.01%、large +0.17%。
- [x] small AP +5%与small AR正提升门均失败；路线淘汰，论文主模型保持官方P2-640。
- [x] 报告：`results/reports/bdd-modules/p2_channel_attention_final_report.md`；指标：`results/channel-attention-final/bdd100k-p2-channel-attention-e200-size-metrics.json`。

### 2.34 VisDrone正式协议冻结

- [x] 复核train/val/test-dev为6,471/548/1,610图和343,201/38,759/75,102实例，零空标签、零格式错误。
- [x] val/test-dev small实例占68.59%/67.69%；保存全量尺寸和分类别统计。
- [x] 明确BDD与VisDrone类别语义不同，禁止加载BDD十类检测头；两组均从原始COCO`yolo26n.pt`开始。
- [x] 初始batch32 P2预检OOM；按冻结规则将原生/P2正式配置统一改为batch16/nbs64。
- [x] test-dev冻结为最终一次性测试集；官方评估链完成前禁止启动长训练。
- [x] 原生/P2正式配置均通过受控preflight；数据哈希一致、run目录独立、GPU0可用。
- [x] 协议：`results/reports/visdrone/visdrone_protocol_freeze.md`；统计：`results/dataset_stats/visdrone/`。

### 2.35 VisDrone官方评估链

- [x] 恢复val/test-dev官方annotations 548/1,610份并记录树哈希；原ZIP保持不变。
- [x] 实现官方8列预测导出：原图XYWH、类1–10、score全局降序、maxDets 500。
- [x] 移植官方v1.0.4 ignored-region、匹配、VOC AP和AR逻辑；服务器无MATLAB，明确标注为算法兼容Python端口。
- [x] 4个合成测试及548图/247,384检测真实格式烟测通过。
- [x] 报告：`results/reports/visdrone/visdrone_official_eval_prep.md`。

### 2.36 VisDrone原生/P2配对显存预检

- [x] batch32原生约16.5 GB通过；P2约22.3 GB触发assigner OOM/CPU回退，正式判定失败。
- [x] 两组统一改为batch16/nbs64并创建新的run名称，禁止在失败run中混改batch。
- [x] batch16原生约9.03 GB/41.98秒，P2约22.7 GB/88.37秒；CSV有限且无OOM/回退。
- [x] 冻结100-epoch调度、epoch20阶段门、seed0与其余公平配置；报告：`results/reports/visdrone/visdrone_paired_preflight_report.md`。

### 2.37 VisDrone原生/P2 Stage 20与关闭

- [x] 两组按相同100-epoch调度完成20/20并健康暂停；`last.pt`均保留EMA、optimizer和scaler，可精确续训。
- [x] 独立val、COCO尺寸和官方ignored-region兼容评估全部完成；所有评估引用固定`best.pt`哈希。
- [x] P2相对原生control：all AP -1.47%、small AP +3.11%、small AR -6.92%、medium AP -2.47%、large AP -13.82%。
- [x] 官方兼容AP50-95为15.5235% vs 15.2394%（+1.86%），但不推翻预注册尺寸门；P2不续训100且不进入test-dev。
- [x] 完整报告：`results/reports/visdrone/visdrone_stage20_report.md`；原始指标：`results/visdrone-stage20/`。

### 2.38 VisDrone原生e100最终训练与评估

- [x] 原生control从epoch20精确恢复并正常完成100/100；最佳epoch72 mAP50-95 0.1757，无early stopping/OOM/NaN/CPU回退。
- [x] 最终val尺寸AP all/small/medium/large为0.180351/0.098479/0.267417/0.427754；官方兼容AP50-95为18.9947%。
- [x] 模型冻结后完成唯一一次test-dev：尺寸AP为0.143200/0.064963/0.220325/0.329504；官方兼容AP50-95为18.3414%。
- [x] test-dev从现在起永久封存，不再用于训练、调参、选模或额外seed评估。
- [x] 报告：`results/reports/visdrone/visdrone_native_e100_final_report.md`；原始指标：`results/visdrone-native-e100/`。

### 2.39 BDD冻结预测定性误差分析

- [x] 预测SHA256与配对Bootstrap冻结输入一致；未训练、未重新推理、未访问VisDrone test-dev。
- [x] 固定置信度0.25、同类一对一IoU 0.50和small面积小于32²像素。
- [x] 完整候选：P2修复small漏检6,010个、共同small漏检52,563个、P2新增误检12,948个。
- [x] 三类候选先写入JSONL，再按固定规则各取前三个不同图像；未看图后替换。
- [x] 三组GT/YOLO26n/P2并列PNG/PDF、manifest和报告已落盘。
- [x] 报告：`results/reports/analysis/bdd_error_analysis_report.md`；资产：`results/bdd-error-analysis/`。

### 2.40 P2到P3小目标蒸馏工程验收

- [x] 在`feat/p4-p2p3-distillation`显式分离学生`[16,19,22,23]`与教师`[19,22,25,28,29]`的特征/检测头hook。
- [x] 实现D1公共P3/P4/P5映射、D2跨层`concat(avg_pool(T2),T3)`与连续small掩码、D3双分支几何均值可靠性。
- [x] CPU FP32和RTX 3090 CUDA/FP16完整损失反向通过；教师无梯度，学生与投影器梯度有限。
- [x] 深拷贝恢复和student-only剥离通过；最终best/last不含教师或投影器。
- [x] 固定400/100、640、batch16/nbs64、3 epoch smoke完成；峰值日志3.03 GB，无OOM、CPU回退、NaN或Inf。
- [x] `train/dis_loss`按epoch为0.15537/0.12793/0.10662；只作健康证据，不作精度筛选。
- [x] 首次启动因worktree缺少Ultralytics AMP自检图片在首批前终止；失败元数据已归档，补接只读官方asset后完成，无训练状态混用。
- [x] 最终学生为2,507,700参数、5.791 GFLOPs、stride `[8,16,32]`；best SHA256为`dfd8ea90a88d26cf1e2b7cb58ac4f919573ae9dab346e9e5e5462febff955400`。
- [x] 规范化run：`runs/detect/phase4-distill-smoke/bdd100k-yolo26n-p2p3-consensus-640-e3-b16-seed0/`。

### 2.41 P2到P3小目标蒸馏Stage20与路线关闭

- [x] 提交`c927ca2`冻结D0-D3；全部使用BDD 70k/10k、640、batch16/nbs64、seed0和200轮调度，并在epoch20保存完整状态后暂停。
- [x] 四组均为20/20，optimizer/scaler/EMA完整；总矩阵约9.97小时，无OOM、CPU回退、NaN、Inf或自动fitness恢复。
- [x] D0尺寸指标：all/small/medium/large AP为0.226392/0.073483/0.252215/0.482080，small AR为0.205715。
- [x] D1相对D0：all AP -3.78%、small AP -7.26%、small AR -5.24%，停止。
- [x] D2相对D0：all AP +0.18%、small AP +1.43%、small AR +0.89%，未达到small AP +5%门，停止。
- [x] D3相对D0：all AP +0.10%、small AP +1.42%、small AR +0.43%，未达到门且未优于D2，停止。
- [x] D1-D3均不进入Stage40或200轮，不调`dis`、面积常数、背景下限或追加新变体。
- [x] 报告：`results/reports/original-methods/p2p3_stage20_report.md`；机器结果：`results/p2p3-stage20/summary.json`。

### 2.42 RDCI工程验收与Stage20准备

- [x] 审计DCAS论文和GPL-3.0官方代码；只引用P1浅层注入动机，不复制其DSPPF、数据扫描或整网YAML。
- [x] 实现P1对齐、细节/上下文拆分和零初始化P2残差；加载全部官方P2权重后初始预测逐元素一致。
- [x] 参数2,527,760，较官方P2增加9,408（0.37%）；GFLOPs 7.540→8.026（+6.44%），stride仍为`[4,8,16,32]`。
- [x] 53项完整测试通过48项、条件跳过5项；真实RTX 3090 CUDA/FP16反向通过。
- [x] 3 epoch smoke完成，日志峰值14.1 GB、权重有限且残差已学习；小子集低AP触发两次fitness恢复，只作工程记录。
- [x] batch64固定子集预检完成，日志峰值21.8 GB；考虑完整数据余量，两组正式矩阵统一冻结batch32/nbs64。
- [x] 正式R0官方P2/R1 RDCI均使用200轮计划并在epoch20保存完整状态后暂停；报告：`results/reports/original-methods/rdci_prep_report.md`。

### 2.43 RDCI公平Stage20与路线关闭

- [x] R0/R1均完成20轮并按设计暂停；optimizer/scaler/EMA完整，训练用时3.48/3.66小时。
- [x] 两组20行CSV全部有限，无OOM、CPU回退、NaN、Inf或fitness恢复。
- [x] 完整10k尺寸评估R0→R1：all AP 0.235178→0.237999（+1.20%）。
- [x] small AP 0.083358→0.084439（+1.30%），低于+5%门；small AR 0.257471→0.256035（-0.56%），核心门失败。
- [x] medium/large AP分别+1.11%/+1.03%，但不能替代small门；RDCI不进入Stage40/200轮或追加变体。
- [x] 报告：`results/reports/original-methods/rdci_stage20_report.md`；机器结果与哈希：`results/rdci-stage20/`。

### 2.44 PGSR可行性与自适应分辨率路由上限

- [x] 官方P2 e200权重和BDD train哈希固定500图开发清单冻结；正式val未使用。
- [x] B0全图640、B1同权重全图960、B2标准SAHI和B3 GT Oracle Top-2全部完成。
- [x] B1相对B0：all AP +8.46%、small AP +14.83%、small AR +11.72%，分辨率响应成立。
- [x] B2/B3 small AP分别-56.36%/-43.78%，局部裁块路线关闭。
- [x] 25%全图960 GT路由上限：all AP +1.25%、small AP +11.18%、small AR +11.92%，进入可学习路由头准备。
- [x] 报告：`results/reports/original-methods/pgsr_feasibility_report.md`；机器结果：`results/pgsr-feasibility/`。

### 2.45 透视分带路由工程与开发门

- [x] 5,000训练图与500开发图哈希清单重叠为0；训练标签正/零/负2,978/504/1,518。
- [x] 从冻结P2第19层提取4个垂直带的均值/最大值，路由头8,769参数，检测器不反向更新。
- [x] seed0固定25%路由相对B0：all AP +0.14%、small AP +7.27%、small AR +9.89%。
- [x] seed0同时超过随机与粗检small框置信度和启发式；seed1/2 small AP和AR也均为正提升。
- [x] 正式协议与seed0 checkpoint哈希冻结：`docs/protocols/PERSPECTIVE_ROUTER_FINAL_PROTOCOL.md`。

### 2.46 透视分带路由正式10k评估

- [x] 在冻结提交`da998af`后执行全640、全960、学习/启发式/随机25%和GT Oracle 25%。
- [x] 学习路由相对B0：all AP +4.31%、small AP +13.75%、small AR +8.76%、large AP -1.11%。
- [x] 学习路由small AR高于启发式0.003095，但small AP低0.000724；严格比较门失败。
- [x] 当前顺序实现估算98.924秒，高于全960的87.655秒；不宣称加速。
- [x] 报告：`results/reports/original-methods/perspective_router_final_report.md`；机器结果：`results/perspective-router-final/`。

### 2.47 仓库资料整理

- [x] 根目录收敛为README、状态、下一阶段和会话交接四个入口。
- [x] 论文材料移入`docs/paper/`，有效协议移入`docs/protocols/`，失败路线协议移入`docs/archive/protocols/`。
- [x] Markdown实验报告按基线、BDD模块、原创候选、VisDrone和分析归入`results/reports/`；机器证据目录保持原路径。
- [x] 删除两份基于实验前YOLO26-SDF和更早YOLO11假设的失效总方案。
- [x] 新增`docs/REPOSITORY_MAP.md`和`results/README.md`，明确源码、配置、工作树和忽略产物的生命周期。
- [x] 历史工作树尚有未逐文件规范化的忽略产物，本轮不删除。

### 2.48 路由配对Bootstrap

- [x] 使用正式10k冻结预测完成学习路由对全640、对粗检启发式各1,000次同图像配对Bootstrap；未重新训练或推理。
- [x] 对全640的all/small AP差值95%区间为`[+0.00942,+0.01338]`和`[+0.01102,+0.01693]`，均1,000/1,000次为正。
- [x] 对启发式的all/small AP区间为`[-0.00666,+0.00110]`和`[-0.00282,+0.00170]`，均跨0；不支持优越性或等价性。
- [x] 统计补证不改变预注册点估计门失败；报告：`results/reports/analysis/router_paired_bootstrap_report.md`。

### 2.49 集成端到端运行时

- [x] 冻结500图、RTX 3090、FP32、显式batch16、每策略预热和3次轮换重复协议。
- [x] 集成学习头复用本轮640 P2特征；Top125成员集合与冻结正式路由分数一致。
- [x] 全640/全960/随机25%/启发式25%/学习25%总时间为2.8839/3.6136/3.8250/3.7723/3.7802秒。
- [x] 学习25%比全960慢4.61%，吞吐低4.44%；三个25%策略均无速度优势。
- [x] 初始两次执行暴露Ultralytics路径列表会绕过`batch=16`并导致OOM；已显式切块并保留工程记录。
- [x] 报告：`results/reports/analysis/integrated_router_runtime_report.md`；机器结果：`results/integrated-router-runtime/benchmark.json`。

## 3. 尚未完成

### 3.1 尚未开始

- [x] 将P2到P3蒸馏负结果边界并入论文结果/讨论，但不作为有效原创贡献。
- [ ] 获得正式论文源稿后统一章节编号、表格和引用格式。

### 3.2 下一工作包

- [x] 完成Copy-Paste道路场景文献审计，冻结PASCP候选、原创边界和公平Stage20协议。
- [x] 完成57项全量回归和50个真实增强样本可视化抽查，框合法性通过。
- [x] 完成固定子集3 epoch工程smoke；训练链健康，一次小子集fitness恢复已记录。
- [x] 完成R0/R1公平Stage20、可恢复checkpoint和完整10k尺寸评估；PASCP核心门失败。
- [x] 冻结推理期自适应分辨率路由研究问题，并确认GT上限通过。
- [x] 建立互斥5,000图训练集，实现并训练轻量P2路由头；三seed开发门通过。
- [x] 完成一次正式10k val、效率估算和最终判定；不在同一val上重测变体。
- [x] 完成学习路由对全640及粗检启发式的1,000次同图像配对Bootstrap。
- [x] 完成同硬件、同计时边界的集成端到端运行时比较。
- [ ] 在不复用BDD正式val选型的独立数据或新冻结划分上验证路由。
- [ ] P17收口合入`master`后，决定是否按`docs/plans/MODULE_IMPROVEMENT_PROPOSALS.md`启动拟P18候选方案之一；启动前须为该方案单独冻结预注册协议。
- [x] P18B ACR已于2026-08-29通过Stage20预注册继续门（small AP +9.81%）；是否进入Stage40/200轮配对训练待决策。

## 4. 当前风险与决策

| 风险 | 当前状态 | 决策 |
|---|---|---|
| 上游源码没有 Git 元数据 | 未获得上游官方 commit/tag | 项目使用独立 Git commit 冻结完整源码，并继续记录 Ultralytics 版本、权重和源码树哈希 |
| BDD100K 官方下载入口不可用 | 官方门户超时，旧镜像失效 | 已使用可校验镜像；保留原始 ZIP、标注快照、哈希和跨源字节比对证据 |
| BDD100K 标注来自可追溯镜像而非官方 ZIP | 已完成 10 张跨源逐字节核验和全量结构扫描 | 可用于当前实验；官方 ZIP 恢复后再做独立内容复核 |
| P2 可能被重复实现 | 官方已经存在 P2 YAML | 直接复用官方 P2，禁止新增重复模块 |
| 新增模块过早导致消融不可信 | 正式基线和官方 P2 已独立验证 | SDD、DWF、SA-NWD 依次独立实现和验证，禁止一次叠加 |
| P2 计算与显存成本较高 | GFLOPs +29.3%，训练约 2.05 倍，batch 64 峰值接近 24 GB | 保留 P2 的同时，将显存与速度作为 SDD/DWF 设计硬约束；每个模块先做固定 smoke |
| SDD 工程可行但精度退化 | 配对短跑 all AP -49.78%、small AP -61.48%，所有尺寸 AP/AR 下降 | 淘汰 SDD，不做 200 epoch，不与 DWF/SA-NWD 叠加；保留代码与证据供复查 |
| DWF 动态采样反向非位级确定 | PyTorch CUDA grid-sample backward 在 deterministic warn-only 下继续运行 | 固定环境和配对流程；正式结论使用多 seed 均值/标准差，不声称逐位复现 |
| DWF 及两个降级变体均未通过 | 完整、nearest+gate、nearest+weight-only 均整体退化；Variant B all AP -17.41%、small -12.40% | 淘汰 DWF，恢复官方 P2 concat；保留代码与负结果证据，不继续扩展或组合 |
| SA-NWD smoke 触发一次 fitness 自动恢复 | 第 3 epoch 早期指标极低命中 `fitness == 0`；loss/权重均有限，恢复后完成 | 不视为 NaN/OOM，但在 control/candidate 配对筛选中记录恢复次数，若重复超过内置上限则停止并降级 alpha |
| SA-NWD 三组诊断均未通过 | 各变体 small AP 有 2.37%–6.90% 增益，但 all AP 下降 5.28%–6.75%，两组 small AR 下降 | 淘汰 SA-NWD，不做 200 epoch或继续调参；恢复官方 P2，保留默认关闭代码和证据 |
| P2-960 正式训练成本过高 | 22.68 分钟/epoch，200 epoch 约75.6小时；已完成39 epoch | 用户决定正式终止且不恢复；保留checkpoint，只做完整尺寸评估，主模型使用P2-640 |
| 通道注意力Stage 20 batch64显存余量较小 | 小子集预检20.8 GB；完整数据日志22.4 GB、物理约23.8/24.6 GB | 当前稳定则继续监控；若OOM立即停止并重新冻结batch，禁止在同一run自动降batch混跑 |
| 首轮Stage 20学习率协议不公平且不可续 | `epochs=20`导致快速衰减；正常结束剥离optimizer/scaler | 降级为诊断；重新使用官方200-epoch计划，并在checkpoint保存后由启动器强制暂停、保留完整状态 |
| 首次Stage40 resume权重被pretrained覆盖 | epoch21前发现loss重置且只迁移360/904项；已停止，CSV/last未变 | Trainer在resume所有权处强制`pretrained=False`；训练前逐张量验证实时模型等于checkpoint EMA |
| smoke 输出目录发生嵌套 | Phase 0 产物保留原路径 | 正式启动器将 project 转为绝对路径，并在已有目录时拒绝运行 |
| 根目录 `.git` 是只读空挂载 | 普通 Git 命令无法识别仓库 | 使用 `.project.git` 保存真实对象；启动器显式读取该 Git 目录 |
| GPU1 被其他长期任务占用 | `dit_env` diffusion 训练持续占用 GPU1 | 不抢占外部任务；640 基线改用空闲 GPU0，保持总 batch 64 |
| 前台训练会话未持久化 | 约 87 分钟后进程消失；无 OOM/重启/GPU 错误证据 | 已新增 `resume_experiment.py`，使用独立 tmux 会话续训并记录 lock/history |
| tmux 服务器继承 CUDA 映射 | 首次续训的逻辑 GPU0 映射到物理 GPU1；在 epoch 18 未完成前已中止 | 新会话显式设置 `CUDA_VISIBLE_DEVICES=0`，resume history 同步记录该环境变量 |
| BDD 类别极不均衡 | 类别实例差异明显，`train` 类别仅 151 个实例 | 正式基线报告分类别 AP，并在论文中说明长尾问题；暂不擅自重采样 |
| 上游蒸馏默认假设教师/学生层号一致 | 已显式分离原生学生16/19/22/23与P2教师19/22/25/28/29，并通过真实模型测试 | D0-D3只使用已冻结映射；禁止回退到共享层号 |
| 蒸馏方法与ScaleKD/FGD/CrossKD/MSAD存在先验重叠 | 蒸馏、前景掩码和跨尺度对齐本身不是原创 | 原创声明只限P2到P3目标、YOLO26双分支可靠性和零推理增量组合；完整消融前不宣称有效创新 |
| P2到P3蒸馏Stage20收益不足 | D2/D3 small AP仅+1.43%/+1.42%，D3未优于D2；D1明显退化 | 按预注册门关闭路线，不进入Stage40/200轮，不调权或追加变体；作为负结果保留 |
| 教师前向增加训练成本 | D0约2.02小时，D1/D2/D3各约2.59/2.67/2.67小时；最终推理结构不变 | 路线已关闭；保留成本证据，不再消耗训练资源 |
| RDCI与DCAS先验重叠 | P1浅层注入和上下文聚合已有公开先验 | 明确引用DCAS；原创声明只限YOLO26-P2恒等残差细节/上下文组合及受控验证 |
| RDCI小子集fitness恢复 | 3 epoch smoke的epoch2/3因AP近零各恢复一次，loss和权重均有限 | 不用于精度判断；正式完整数据任一组出现自动恢复即停止并审计 |
| RDCI batch64余量过小 | 固定子集日志峰值21.8 GB，完整BDD预计更高 | R0/R1统一batch32/nbs64，禁止只给候选降低batch |
| RDCI Stage20收益不足 | all AP +1.20%，small AP仅+1.30%，small AR -0.56% | 按预注册门关闭，不进入Stage40/200轮，不调核、权重、宽度或位置 |
| PASCP与既有Copy-Paste/透视增强重叠 | 小目标复制、道路上下文和透视约束均有公开先验 | 不主张通用思想原创；只主张纯框同图类别平衡/底边带/遮挡拒绝组合，且正收益前只称候选 |
| “自研模块”可能被限定为推理网络层 | PASCP只在训练时工作，推理图与官方P2完全相同 | PASCP已因精度门失败而关闭；后续候选必须在训练前确认导师认可的创新类型 |
| PASCP Stage20小目标退化 | all AP -1.32%，small AP -3.70%，small AR -6.42% | 按预注册门关闭，不进入Stage40/200轮，不搜索概率、阈值、羽化或流水线位置 |
| 标准SAHI与GT裁块均退化 | B2/B3 small AP -56.36%/-43.78%，即使B3覆盖99.71% small中心仍失败 | 关闭640裁块路线，不搜索切片尺寸、重叠率、融合阈值或Top-K |
| 自适应分辨率仍只有GT上限 | 25% Oracle small AP +11.18%、small AR +11.92%，但推理不可使用GT | 仅允许训练轻量P2路由头；学习路由通过固定开发门前不宣称有效模块 |
| 学习路由可能只复现粗检启发式 | seed0开发small AP/AR 0.123838/0.326433，高于启发式0.122799/0.315192 | 正式val要求small AP和AR均严格高于启发式，否则不作为有效最终方法 |
| 正式val学习头未全面超过启发式 | 学习small AP 0.115461低于启发式0.116185，虽small AR更高 | 按预注册门关闭；保留相对B0正收益，不宣称学习头必要性，不在val上换seed重测 |
| 集成路由没有速度优势 | 学习25%比全960慢4.61%；随机/启发式也慢5.85%/4.39% | 不宣称加速或更优精度-速度前沿；若改执行结构，必须作为新方法在独立数据上预注册 |
| 检测前CNN未超过图像统计基线 | seed0 utility capture 0.36375，固定岭回归统计为0.41027，低11.34% | 按P12预注册门停止；不跑seed1/2、不访问BDD val、不调尺寸、宽度、预算或损失 |

## 5. 下一工作包

`P12-PRE-INFERENCE-UTILITY-ROUTER`已按开发门关闭。工程、缓存、seed0完整历史、同成本基线、哈希和报告位于`results/pre-inference-router-*`及`results/reports/original-methods/pre_inference_utility_router_screen_report.md`。

当前没有获准继续的训练或评估工作包。算法探索已停止，下一步只做SCI目标期刊核验、作者指南映射、开放科学清单、投稿信和学校模板接入；不得绕过P12/P14停止门继续优化路由。

2026-08-27补充：已记录拟P18频域/小波模块改进候选方案（`docs/plans/MODULE_IMPROVEMENT_PROPOSALS.md`），仅为候选记录，不改变上述停止边界；启动前置条件与纪律见该文档及`docs/archive/plans/NEXT_STAGE_PLAN.md`第6节。

当前不得恢复PASCP、RDCI、P2到P3蒸馏、P2-960训练或历史失败模块；VisDrone test-dev继续封存。GT Oracle不得作为实际推理方法。

## 6. 正式实验停止门

在以下条件全部满足前，不启动 200 epoch 训练：

1. [x] `P0-SMOKE` 验收通过。
2. [x] 项目有可引用的 Git commit。
3. [x] 正式数据 YAML 和训练配置冻结。
4. [x] 尺寸专项评估脚本可以运行。
5. [x] run 输出不会覆盖已有结果。

在以下条件全部满足前，不开发 SDD/DWF/SA-NWD：

1. [x] YOLO26n 正式基线完成。
2. [x] 基线日志、权重、整体指标和 AP(small) 均可复查。
3. [x] 官方 P2 已作为第一个独立消融验证。

## 7. 状态同步规则

后续每完成一个工作包，必须在同一次变更中同步：

1. 更新本文件的“最后更新”“当前阶段”“已完成/尚未完成”。
2. 在下方更新日志追加一行，只记录有证据的结果。
3. 环境变化时更新 `environment/conda.yaml` 和 `environment/pip-freeze.txt`。
4. 硬件或 CUDA 变化时更新 `environment/hardware.txt`。
5. YOLO26 接口或源码变化时更新 `environment/yolo26_api_audit.md`。
6. 数据、类别、划分或标签变化时更新 `environment/dataset_audit.md` 和对应 YAML。
7. 训练完成时记录 commit、命令、配置、seed、GPU、权重和原始 run 路径。
8. 没有日志、权重或哈希支撑的指标，不得写入“已完成”或论文结果。

## 8. 更新日志

| 日期 | 工作包 | 状态 | 证据 |
|---|---|---|---|
| 2026-07-25 | P0-ENV | 完成 | Conda 环境、硬件记录、YOLO26/P2 构建、GPU 预测 |
| 2026-07-25 | P0-DATA-BDD | 完成 | 100K 图像、80K YOLO 标签、全量扫描 0 corrupt、哈希审计 |
| 2026-07-25 | P0-DATA-VISDRONE | 完成 | 8,629 图像/标签、全量扫描 0 corrupt、ZIP 与标签树哈希 |
| 2026-07-25 | P0-SMOKE | 完成 | 400/100 固定子集、250 图人工审计、3 epoch train、独立 val/predict、验收报告 |
| 2026-07-25 | P1-BASELINE-PREP | 完成 | 独立 Git、尺寸评估自测、全量统计、正式配置和受控启动器 |
| 2026-07-25 | P1-BASELINE-640 | 中断待续训 | 完成 17/200 epoch、累计 5,224 秒；非早停；checkpoint 可续训 |
| 2026-07-26 | P1-BASELINE-640-CHECK | 完成 | 无进程、无 OOM/重启证据；best=last；中断记录已落盘 |
| 2026-07-26 | P1-BASELINE-640-RESUME | 完成 | 受控续训至 200 epoch，23:08 正常结束，无 early stopping |
| 2026-07-27 | P1-BASELINE-640-EVAL | 完成 | 完整独立 val；small/medium/large AP/Recall；基线验收报告 |
| 2026-07-27 | P1-OFFICIAL-P2-PREP | 完成 | 官方 P2 YAML、正式配置和同源预训练初始化冻结 |
| 2026-07-27 | P1-OFFICIAL-P2-SMOKE | 完成 | batch 64、峰值 20.8 GB、train/val/save 通过、无 NaN/Inf |
| 2026-07-27 | P1-OFFICIAL-P2-TRAIN | 执行中 | epoch 1；物理 GPU0；约 21.4 GB；MuSGD；tmux `yolo26-p2-640` |
| 2026-07-28 | P1-OFFICIAL-P2-TRAIN | 完成 | 200/200 epoch；最佳 epoch 183；无 early stopping、NaN、Inf 或 OOM |
| 2026-07-28 | P1-OFFICIAL-P2-EVAL | 完成 | 完整独立 val；small AP50-95 0.101480，较基线 +14.32%；P2 验收通过 |
| 2026-07-28 | P2-SDD-IMPLEMENT | 完成 | 仅 P2 SDD；FP32/FP16、梯度、序列化、YAML、TorchScript 与项目外加载通过 |
| 2026-07-28 | P2-SDD-SMOKE | 完成 | 400/100、3 epoch、batch 64；最高约 22.3 GB；无 NaN/Inf/OOM；不作精度结论 |
| 2026-07-28 | P2-SDD-SCREEN | 完成/淘汰 | 配对 50 epoch；all AP -49.78%、small AP -61.48%；不进入正式训练或后续组合 |
| 2026-07-29 | P2-DWF-IMPLEMENT | 完成 | 动态采样、权重、门控、FP32/FP16、序列化、YAML 与 TorchScript 通过 |
| 2026-07-29 | P2-DWF-SMOKE | 完成 | 400/100、3 epoch、batch 32；最高约 14.6 GB；无 NaN/Inf/OOM；不作精度结论 |
| 2026-07-29 | P2-DWF-SCREEN | 完成/待诊断 | 配对 50 epoch；all AP -19.78%、small AP -2.41%；进入 nearest/无门控降级诊断 |
| 2026-07-29 | P2-DWF-DIAG-A-PREP | 完成 | nearest+权重+门控；CPU/CUDA/AMP/YAML/重载与 3 epoch smoke 通过；准备配对筛选 |
| 2026-07-29 | P2-DWF-DIAG-A-SCREEN | 完成/未通过 | small AP +14.87%，但 all AP -15.01%；不做 200 epoch，继续去门控 Variant B |
| 2026-07-29 | P2-DWF-DIAG-B-PREP | 完成 | nearest+weight-only；CPU/CUDA/AMP/YAML/重载与 3 epoch smoke 通过；准备最终筛选 |
| 2026-07-29 | P2-DWF-DIAG-B-SCREEN | 完成/淘汰 | all AP -17.41%、small AP -12.40%；DWF 全路线终止，恢复官方 P2，进入 SA-NWD |
| 2026-07-29 | P2-SA-NWD-PREP | 完成 | 双分支、原生关闭等价、CUDA/AMP、640/960 映射与 3 epoch smoke 通过；准备配对筛选 |
| 2026-07-29 | P2-SA-NWD-SCREEN | 完成/待降级 | small AP +6.90%、AR +1.60%，但 all AP -5.28%；继续 small-only 诊断 |
| 2026-07-29 | P2-SA-NWD-SMALL-ONLY | 完成/未通过 | small AP +3.43%、AR -2.02%、all AP -6.75%；进入 τ=512 最终诊断 |
| 2026-07-29 | P2-SA-NWD-TAU512 | 完成/淘汰 | small AP +2.37%、AR -0.36%、all AP -6.52%；SA-NWD 路线终止，恢复官方 P2 |
| 2026-07-29 | P2-RESOLUTION-PREP | 完成 | P2-960、batch16、3 epoch smoke 通过；峰值约 14.7 GB；640/960 配置冻结 |
| 2026-07-29 | P2-RESOLUTION-SCREEN | 完成/通过 | all AP +40.49%、small AP +102.58%、small AR +56.61%；进入完整 200 epoch |
| 2026-07-29 | P2-RESOLUTION-960-FULL | 执行中 | 19:27 启动；tmux `yolo26-p2-960-full`；物理 GPU0；初始采样约 19.2 GB |
| 2026-07-30 | P2-RESOLUTION-960-FULL-CHECK | 健康执行中 | 38/200；14.36 h；best mAP50-95 0.28471；CSV 有限；GPU0 约 18.3 GB/98% |
| 2026-07-30 | P2-RESOLUTION-960-PAUSE | 人工暂停 | 39/200；best mAP50-95 0.28507；checkpoint 可精确恢复；GPU0 已释放；等待评估/恢复决策 |
| 2026-07-30 | P2-RESOLUTION-960-STOP-DECISION | 正式终止 | 用户决定不恢复200 epoch；主模型保持P2-640；下一步仅做epoch39完整评估 |
| 2026-07-30 | P2-RESOLUTION-960-CLOSEOUT | 完成/关闭 | 10k small AP +19.46%、small AR +9.83%、all AP +6.11%；不恢复训练，转浅层注意力 |
| 2026-07-30 | P2-EARLY-CHANNEL-ATTENTION-PREP | 完成/通过 | +4,160参数；CUDA/FP16/YAML/重载和3 epoch smoke通过；进入配对50 epoch筛选 |
| 2026-07-30 | P2-EARLY-CHANNEL-ATTENTION-SCREEN | 完成/通过 | small AP +9.64%、small AR +1.60%、all AP +6.95%；全部尺寸门通过，进入20 epoch阶段门准备 |
| 2026-07-30 | P2-EARLY-CHANNEL-ATTENTION-STAGE20-PREP | 完成/通过 | batch64预检峰值20.8 GB；完整BDD 20 epoch配置和worktree正式启动器已冻结 |
| 2026-07-30 | P2-EARLY-CHANNEL-ATTENTION-STAGE20 | 执行中 | 16:24启动；完整70k/10k扫描0 corrupt；epoch 1日志显存22.4 GB且稳定；tmux `yolo26-p2-attn-stage20` |
| 2026-07-30 | P2-EARLY-CHANNEL-ATTENTION-STAGE20-DIAG | 完成/降级 | 20/20；mAP50-95 0.23881；健康但学习率计划不公平且checkpoint不可续，仅作诊断 |
| 2026-07-30 | P2-EARLY-CHANNEL-ATTENTION-FAIR-STAGE20 | 执行中 | 21:59启动；200-epoch同调度、epoch20强制暂停；tmux `yolo26-p2-attn-fair-stage20` |
| 2026-07-31 | P2-EARLY-CHANNEL-ATTENTION-FAIR-STAGE20 | 完成/通过 | 01:31暂停；mAP50-95较官方同epoch-0.46%；checkpoint完整，允许精确恢复至epoch40 |
| 2026-07-31 | P2-EARLY-CHANNEL-ATTENTION-STAGE40-RESUME-AUDIT | 安全中止 | epoch21完成前发现pretrained覆盖EMA；CSV仍20行、last哈希未变；进入resume修复验证 |
| 2026-07-31 | P2-EARLY-CHANNEL-ATTENTION-STAGE40-RESUME-FIX | 完成/通过 | 904/904迁移、904张量精确匹配；epoch2连续训练并再次可恢复暂停 |
| 2026-07-31 | P2-EARLY-CHANNEL-ATTENTION-FAIR-STAGE40 | 执行中 | 08:53精确恢复epoch21；904张量匹配EMA、loss连续；tmux `yolo26-p2-attn-fair-stage40` |
| 2026-07-31 | P2-EARLY-CHANNEL-ATTENTION-FAIR-STAGE40 | 完成/通过 | 12:27暂停；mAP50-95较官方同epoch-0.98%；small AP/AR较Stage20 +3.05%/+2.75%，允许至Stage60 |
| 2026-07-31 | P2-EARLY-CHANNEL-ATTENTION-FAIR-STAGE60 | 执行中 | 13:32精确恢复epoch41；904/904迁移、904张量匹配EMA，初始loss连续；tmux `yolo26-p2-attn-fair-stage60` |
| 2026-07-31 | P2-EARLY-CHANNEL-ATTENTION-FAIR-STAGE60 | 完成/通过 | 17:03暂停；mAP50-95较官方同epoch-0.35%，epoch40→60 +0.00671，checkpoint完整 |
| 2026-08-02 | P2-EARLY-CHANNEL-ATTENTION-FULL-COMPLETION | 执行中 | 08:40从epoch61精确恢复；904张量匹配EMA、loss连续；不设人工stop epoch |
| 2026-08-03 | P2-EARLY-CHANNEL-ATTENTION-FULL-COMPLETION | 完成 | 200/200；最佳epoch190 mAP50-95 0.26904；无early stopping/OOM/NaN |
| 2026-08-04 | P2-EARLY-CHANNEL-ATTENTION-FINAL-EVAL | 完成/淘汰 | small AP +0.14%、small AR -1.70%、medium AP -4.95%；保留门失败，恢复官方P2-640 |
| 2026-08-04 | VISDRONE-PROTOCOL-FREEZE | 完成 | 原生/P2配对；COCO同源初始化；640/b32/e100/stop20；test-dev最终一次性评估 |
| 2026-08-04 | VISDRONE-CONFIG-PREFLIGHT | 完成 | 两组配置和数据哈希通过；GPU0可用；正式run目录互不冲突且未创建 |
| 2026-08-04 | VISDRONE-OFFICIAL-EVAL-PREP | 完成 | annotations恢复；4个合成测试+548图格式烟测通过；兼容端口可执行 |
| 2026-08-04 | VISDRONE-PAIRED-PREFLIGHT-B32 | 失败/降级 | 原生约16.5 GB通过；P2约22.3 GB触发assigner OOM并回退CPU；两组统一降batch16 |
| 2026-08-04 | VISDRONE-PAIRED-PREFLIGHT-B16 | 完成/通过 | 原生9.03 GB/41.98秒；P2 22.7 GB/88.37秒；无OOM/CPU回退，允许Stage20 |
| 2026-08-04 | VISDRONE-STAGE20-PREFLIGHT | 完成 | commit 175f6f4；两组正式b16配置、数据哈希、GPU0和独立run目录全部通过 |
| 2026-08-04 | VISDRONE-PAIRED-STAGE20 | 完成/淘汰P2 | 两组20/20健康暂停且checkpoint可续；P2 small AP +3.11%、AR -6.92%、large AP -13.82%；官方兼容AP +1.86%；不续训100 |
| 2026-08-04 | VISDRONE-CLASS-WEIGHT-PREP | 已冻结 | 原生YOLO26n仅增加cls_pw=0.25；最大权重比2.585；无参数/GFLOPs/推理增量；进入测试和1 epoch预检 |
| 2026-08-04 | VISDRONE-CLASS-WEIGHT-PREFLIGHT | 完成/通过 | 28项测试通过；全量1 epoch约9.03 GB/41.98秒；权重匹配；无OOM/CPU回退/NaN；允许Stage20 |
| 2026-08-04 | VISDRONE-CLASS-WEIGHT-STAGE20 | 完成/淘汰 | all AP +2.97%，small AP +2.42%、small AR -4.69%；官方AP +2.88%但AR500 -1.57%；不续训100 |
| 2026-08-04 | VISDRONE-NATIVE-E100-RESUME | 执行中 | commit e542333；completed20/next21/planned100；708/708迁移与708张量匹配EMA；tmux `visdrone-yolo26n-native-e100-resume` |
| 2026-08-04 | VISDRONE-NATIVE-E100-FINAL | 完成 | 100/100；best epoch72；val尺寸AP 0.18035、官方AP 18.9947%；test-dev尺寸AP 0.14320、官方AP 18.3414%；test永久封存 |
| 2026-08-04 | RESULTS-AND-VISDRONE-MULTISEED-PREP | 完成 | `docs/paper/RESULTS.md`主表；seed1/2仅改seed/name；约2.2小时；只train/val，禁止test-dev |
| 2026-08-05 | VISDRONE-NATIVE-MULTISEED | 完成 | 三seed all AP 0.17843±0.00179、small AP 0.09834±0.00067、官方AP 18.92195±0.06335%；test未访问 |
| 2026-08-05 | PAPER-FIGURES-AND-RESULTS-DRAFT | 完成 | 5组PNG/PDF及哈希清单；BDD/VisDrone误差分析与结果章节初稿；仅读取冻结证据 |
| 2026-08-05 | PAPER-FINAL-MATERIALS | 完成 | 方法/实验协议/正式图注、10条核验引用、自动数值与图文件审计；未训练或访问test-dev |
| 2026-08-05 | BDD-PAIRED-BOOTSTRAP | 完成 | 10k图配对1,000次；all增量CI [+0.00998,+0.01563]、small [+0.00992,+0.01621]；未训练/推理 |
| 2026-08-05 | BDD-QUALITATIVE-ERROR-ANALYSIS | 完成 | 冻结预测10k图；修复/共同漏检/新增误检6,010/52,563/12,948；完整候选、manifest及3组PNG/PDF |
| 2026-08-05 | P4-P2P3-DISTILL-PLAN | 完成/待实现 | 冻结P2教师到原生学生的跨层small蒸馏、双分支可靠性、D0-D3矩阵及20/40/60停止门；未训练 |
| 2026-08-05 | P4-P2P3-DISTILL-PREP | 完成/通过 | run提交e3549b2；CPU/CUDA/AMP/梯度/剥离通过；三轮smoke峰值3.03 GB；最终学生2,507,700参数、5.791 GFLOPs |
| 2026-08-06 | P4-P2P3-DISTILL-STAGE20 | 完成/淘汰 | D1退化；D2/D3 small AP仅+1.43%/+1.42%，低于5%门；全部停止，不进入Stage40/200轮 |
| 2026-08-06 | P5-RDCI-PREP | 完成/通过 | DCAS审计；RDCI恒等迁移、53项回归、CUDA/AMP、3轮smoke和b64预检通过；正式R0/R1统一b32待启动 |
| 2026-08-07 | P5-RDCI-STAGE20 | 完成/淘汰 | all AP +1.20%，small AP仅+1.30%、small AR -0.56%；核心门失败，不进入Stage40/200轮 |
| 2026-08-07 | P6-PASCP-PREP | 完成/通过 | 三篇直接相关论文与官方增强入口审计；纯框同图PASCP、单测、R0/R1配置和停止门冻结；尚无精度结论 |
| 2026-08-07 | P6-PASCP-SMOKE | 完成/通过 | 57项回归、50样本抽查、3轮训练全部完成；峰值14.5 GB；一次小子集fitness恢复已记录；允许Stage20 preflight |
| 2026-08-07 | P6-PASCP-STAGE20 | 完成/淘汰 | all AP -1.32%，small AP -3.70%、small AR -6.42%；核心门失败，不进入Stage40/200轮 |
| 2026-08-07 | P7-PGSR-FEASIBILITY | 完成/裁块淘汰 | B1全图960 small AP +14.83%；B2/B3裁块small AP -56.36%/-43.78%，裁块路线关闭 |
| 2026-08-07 | P7-ADAPTIVE-RESOLUTION-ORACLE | 完成/通过上限门 | 25% GT路由all AP +1.25%、small AP +11.18%、small AR +11.92%；允许准备可学习P2路由头 |
| 2026-08-07 | P7-PERSPECTIVE-ROUTER-DEV | 完成/通过 | 8,769参数；seed0 small AP +7.27%、small AR +9.89%、all AP +0.14%；三seed均通过，正式val协议冻结 |
| 2026-08-07 | P7-PERSPECTIVE-ROUTER-FINAL | 完成/比较门失败 | 正式small AP +13.75%、small AR +8.76%、all AP +4.31%；small AP略低于启发式，按冻结门不保留 |
| 2026-08-08 | P8-REPOSITORY-CONSOLIDATION | 完成 | 删除2份失效总方案；文档与报告分类；71项链接审计；历史工作树保留边界冻结 |
| 2026-08-08 | P8-ROUTER-PAIRED-BOOTSTRAP | 完成 | 各1,000次；对B0的all/small AP区间全正；对启发式区间跨0；不改变预注册门 |
| 2026-08-08 | P9-INTEGRATED-ROUTER-RUNTIME | 完成/无速度优势 | 500图×3；学习25% 3.7802±0.0114秒，比全960慢4.61%；选图集合一致 |
| 2026-08-08 | P11-THESIS-INTEGRATION | 内容完成/待模板 | 中英文摘要、完整无编号正文、模板接入清单、数值/引用/边界回归审计；未启动训练或评估 |
| 2026-08-08 | P12-PRE-INFERENCE-UTILITY-ROUTER | 完成/开发门失败 | 23,417参数；seed0 capture 0.36375低于岭回归0.41027；未跑seed1/2、BDD val、运行时或外部数据 |
