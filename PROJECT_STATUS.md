> **公开快照说明(2026-10-22)**:本文件是内部开发过程日志的脱敏快照,含大量阶段性结论,其中"最终方法"等表述为当时口径。
> 论文 v2 定稿口径以 [README.md](README.md) 与 [docs/paper/THESIS_RESULTS.md](docs/paper/THESIS_RESULTS.md) 为准:**最终模型 ACF-YOLO-RFB**;知识蒸馏(ACF-YOLO-KD)为第五章方法,因组合实验未保持小目标增益而未选入最终部署;RFB 为最终模型组成部分。

# YOLO26-SDF 项目状态

> 本文件是项目当前状态的唯一入口（single source of truth）。
> 最后更新：2026-10-18，Asia/Shanghai
> 当前阶段：**服务器实验侧全部收口（10-17）：泛化工单 P32 VisDrone / P33 KITTI / P34 RTTS（ACDC 经用户确认换源）零样本三集七模型全表入库，k1-rfb 三集 all 维均为七模型最优（RTTS 四维全第一）；Jetson AGX Orin Super 首轮重测完成（旧 Orin NX 数字作废，RFB 5.164ms fp32-IO 口径为表 6-2 依据）；P31 已结案（最终部署模型=L0+RFB 无 KD，all 0.290052 项目历史最高）。双卡空闲无在跑实验。论文正文 v2 在本地 `<project>\01_大论文\`（10-17 序号化重构，旧`论文重写_20260927`路径作废）。剩余：论文侧（泛化章成表/图4-4/docx整册/中期v3）、导师追认 P25–P31+Jetson 口径、异地备份重打（紧急）**
> 新会话交接：`SESSION_HANDOFF.md`

## 1. 当前结论

YOLO26n 640正式基线和官方P2-640独立消融均已验收。P18封箱后BDD当前最优模型为ACR（对官方P2-640：all +4.57%、small AP +9.17%，200轮、三seed与Bootstrap闭环，参数-6.45%且推理更快）；官方P2-640保留为论文对照基线。更早的8,769参数透视分带路由在正式10k val上相对全640取得all AP +4.31%、small AP +13.75%、small AR +8.76%，第一次形成自研模块相对主模型的完整数据正收益。1,000次配对Bootstrap中，all/small AP差值95%区间分别为[+0.00942,+0.01338]和[+0.01102,+0.01693]。但其small AP点估计0.115461略低于等预算粗检启发式0.116185，违反预注册严格门；两者all/small AP差值区间均跨0，不能证明学习头更优。集成运行时中学习25%为3.7802±0.0114秒，比全960的3.6136±0.0868秒慢4.61%，也不具备效率优势。因此它仍是正向精度候选而非已证明必要、实用的最终原创方法。VisDrone test-dev继续永久封存。

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

P18B 200轮全训闭环（2026-09-01）：**ACR路线以完整三级证据链闭环为正向自研结构改动**。(1) 同协议配对终评（R0-200亦完成）：all AP 0.280471 vs 0.269294（+4.15%）、small AP 0.110785 vs 0.103266（**+7.28%**）、small AR +6.99%、large +1.35%、medium -1.98%（如实记录的代价）；配对Bootstrap all/small差值CI `[+0.00848,+0.01338]`/`[+0.00470,+0.01062]`均严格>0。(2) 对官方b64主模型：all +4.57%、small +9.17%，Bootstrap CI亦严格>0。(3) 增益轨迹+9.81%(20轮)→+9.16%(40轮)→+7.28%(200轮)稳定。全部以-6.45%参数（2,490,624 vs 2,662,400）达成，GMACs +17.08%。完整报告：`results/reports/bdd-modules/acr_stage20_report.md` §7-8；机器证据：`results/acr-stage20/`（含full200_summary.json、三份Bootstrap）。

P19A ACFR（2026-09-08/09，用户授权的筛选级证据，进行中）：在冻结ACR上仅将L18 Concat替换为ACFRConcat（两路各50%通道零初始化深度可分离残差细化+联合门控G=4分组互补增益，初始与Concat逐位一致；+8,132参数/+0.33%、+0.211 GMACs/+3.8%）。工程smoke通过（`41406dc`）后Stage20**通过预注册继续门**：all AP 0.248070（+0.83%）、**small AP 0.097757（+6.80%，超+3.0%门两倍余量）**、small AR 0.279380（+0.15%）；相对R0同期累计small AP +17.3%（ACR单独+9.8%），20轮尺度上两模块呈叠加效应（`be4a1d6`、报告`e55f52b`）。epoch40确认训练中（2026-09-09 05:33起GPU0，过夜预授权）。同日下午状态：用户09:13重启的supervisor在10:03 ep40暂停后自动越界续跑200轮（10:04起），**epoch40尺寸评估与确认门被跳过**（偏差已补记协议§7；在线弱证据ep40 +1.64%）。

P19A 200轮终评**通过决定性证据**（2026-09-10）：200/200于18:34完成，best.pt（epoch193）10k尺寸评估all AP 0.282295（+0.65%）、**small AP 0.116565（+5.22%）**、small AR 0.312877（+1.58%）、medium -0.10%、large -0.95%；1,000次同图配对Bootstrap：**small AP差值CI [+0.003070,+0.008839]严格>0**，all AP差值CI [-0.000556,+0.005218]跨0（写平价不写显著）。增益轨迹+6.80%(20轮)→+5.22%(200轮)与ACR自身形态一致；相对R0-200累计small +12.88%、相对官方P2全训+14.86%。在线全尺寸领先从+1.4%收窄至+0.2%但small AP保持，再次印证"在线悲观代理"模式。剩余（需用户授权）：组件消融、seed1/2复制、TensorRT成本含ACR补测；导师追认含ep40跳过偏差。证据：`results/acfr-stage20/full200_summary.json`、`bootstrap-acfr-vs-acr-ep200.json`；报告§5。恢复事故说明：05:38"无声崩溃"系误判——第二次恢复尝试因首次进程仍持有22.62GB显存而OOM退出，首次进程自05:33连续训练至今（机器证据`resume_history.json`，报告§2已更正）。协议`docs/protocols/ACFR_STAGE20_PROTOCOL.md`，报告`results/reports/bdd-modules/acfr_stage20_report.md`；机器证据`results/acfr-stage20/`。另经用户授权在GPU1补测ACR batch1延迟：5.855 vs 6.155 ms/图（-4.87%）。

P19A 组件消融与三seed复制（2026-09-13/15，全部收口）：三seed链09-13闭环（seed0/1/2 small AP +5.22%/+6.21%/+4.72%，均值0.116747±0.000840即+5.38%±0.76%，全部过≥+2.6%跨seed门；`three_seed_summary.json`）。组件消融双臂09-15收口：nr（gate-only）small AP +3.50%（Bootstrap CI [+0.001177,+0.006674]严格>0）、ng（refine-only）+1.74%（CI跨0）；判定**门控必要且单独显著、细化非必要且单独不显著、两组件近似线性叠加**，归因向门控主导调整。执行偏差如实记录（09-14重启、09-15 ng被杀，均检查点精确恢复）。证据：`ablation_summary.json`、`bootstrap-{nr,ng}-vs-acr-ep200.json`；报告§6。TensorRT FP16成本09-15收口（独立环境`yolo26-trt`，冻结环境零改动，`environment/tensorrt_env_audit.md`）：纯engine口径batch1 -0.48%（平价）、batch16 **-8.52%**（ACFR更快，三run稳定），FP16 engine输出过sanity；证据`inference_trt_fp16_gpu0_batch{1,16}.json`，报告§7。**规格§12.3计算类工作全部完成**；剩导师追认（含ep40跳过偏差）与论文写作。

P20 第五章轻量化（2026-09-16/17，按`服务器AI后续实验执行说明_优先版.md`执行，分支`p20-lightweight`）：教师L0复评与冻结值逐项差异=0（10k全量）；逐层成本审计揭示成本地图——Detect头占38.7% MACs不可动、四处高层朴素Conv（L5/L7/L23/L26）参数占25.1%但MACs仅8.4%、高层C3k2块占37.1%参数+13.0% MACs却被候选规则排除。单元筛选U1（DWConv，参数-24.87%/GMACs-8.07%）与U2（GhostConv，-12.25%/-3.97%）Stage20完整10k评估：**small AP两单元均上升（+0.34%/+0.90%，小目标保护设计成功），但all AP -4.45%/-2.60%均未过≤2%门**，U2另有GMACs门预判负——**两单元均败，按冻结协议§7.4第五章结构搜索负结果待裁决**（A接受负结果/B协议修正继续，方向分析见`results/reports/lightweight/unit_failure_analysis_and_directions.md`）。U1另发现TRT 11.3对其L7分组卷积无FP16核（混合精度可部署）。位置消融按用户裁决C在途：P-N（仅neck）GPU1训练中、P-B（仅backbone）暂停于ep6晚间恢复（用户两次让卡给ComfyUI，偏差已记录）。同日完成ACR/ACFR论文补图26张（五列对比/事件案例/门控热图，冻结规则选图，`results/paper-visuals/acr-acfr/`）与P17 UAVDT只读审计（**判定性能失败非工程失败**：Gate C仅败给detection-count启发式0.92%相对差距，同时以CI>0击败全部学习型对照；`results/reports/external/p17_uavdt_closure_audit.md`）。协议`docs/protocols/LIGHTWEIGHT_UNIT_STAGE20_PROTOCOL.md`、`LIGHTWEIGHT_POSITION_ABLATION_PROTOCOL.md`；报告`results/reports/lightweight/lightweight_unit_stage20_report.md`；机器证据`results/lightweight-unit-screen/`、`results/lightweight-position-ablation/`。

**P22 新第五章主线（2026-09-18 06:40冻结，用户携咨询包外部评议后裁定）**：题目改为"基于小目标感知知识蒸馏的轻量化道路目标检测算法"。路线：高容量教师YOLO26s-P2（官方scale s：9,765,856参数/13.9 GMACs≈L0的2.4×；COCO yolo26s.pt SHA256入库）→冻结L0学生（参数/GMACs/延迟不变）→K1输出蒸馏/K2 P2-P3前景加权特征蒸馏/K3联合三臂。教师Stage20须all AP≥+5%且small AP≥+5%；K臂晋升门为all AP≥+1%且small AP≥+2%，或all AP≥-0.5%且small AP≥+3%。**执行进展（2026-09-18）：T0教师Stage20于12:32链完成并大幅过门——all AP 0.289594（+16.74%）、small AP 0.118607（+21.33%）vs L0@ep20冻结值（0.248070/0.097757），双门均超三倍余量；证据`results/kd-teacher/t0-size-val-ep20.json`（`c98b8f9`）。200轮续训已在GPU1恢复（tmux `p22_t200`，~14分/轮，预计周日晨出数，链尾自动10k评估提交`t0-size-val-ep200.json`）。YOLO11n基线因用户让卡暂停于ep153（08:31，第三次GPU0让卡，偏差记入协议§9），已于20:17恢复（tmux `p21_resume`）。**协议`docs/protocols/SMALL_OBJECT_KD_PROTOCOL.md`。

**P23 混同消除对照与MGD蒸馏（2026-09-20晚用户授权启动，2026-09-21裁决落定，2026-09-23终评完成，导师追认待办；协议`docs/protocols/CONFOUND_CONTROL_AND_MGD_KD_PROTOCOL.md`）**：P22三臂全部未过冻结门后授权续研。**裁决（09-21 04:55）：C-b16无蒸馏对照@ep20 = all 0.245575/small 0.097235（vs L0@ep20-b32：all -1.01%/small -0.53%，混同假设成立——b16口径代价恰为P22惜败量级）；同口径重判（门值不变）：K1输出蒸馏 all +2.45%/small +2.28%过门1（四臂唯一）；K2 +0.19%/+0.10%、K3 +2.25%/+0.49%、M1-MGD +2.09%/+0.16%未过。机制结论：输出蒸馏=唯一有效迁移通路，特征蒸馏（前景L1与MGD两种形式）对small零迁移。**终评（09-23）：K1-200（all 0.287017/small 0.116694，`810fc83`）与C-b16-200（all 0.282324/small 0.118558，`21c818f`）双链完成（期间各有用户让卡暂停，已按协议记录）；1000次同图配对Bootstrap（`3a270de`）：主对照K1 vs C-b16 **all +1.66%（+0.47pp，CI95[+0.23,+0.71]pp，P(Δ>0)=1.000统计显著）/small -1.57%（CI95[-0.53,+0.12]pp跨零，P=0.117无显著差异）**；b32参照vs L0-200：all +1.67%（显著）/small +0.11%（无差异）。训练曲线佐证all优势ep20→ep200全程稳定（+0.45~0.8pp），ep20的small优势随对照充分训练收敛消失。复杂度核验：L0/C-b16/K1三模型均为340层/2,417,428参数/10.3 GFLOPs逐项一致；K1与C-b16同设置ONNX导出均506节点、算子构成完全一致，图中无教师/适配器/蒸馏模块。**5.4定性调整：正结果=统计显著的整体AP提升且零复杂度代价，small维度无显著增益；ep20过门→ep200收敛现象本身写入5.3/5.4（叙事变更列入导师追认）**。禁扫超参、禁第三臂（LD/DKD明确出局）。P22证据与5.2/5.3负结果叙事只读保留。

**P24 第六章部署收官（2026-09-24/25，Jetson Orin NX 16GB实机 + RTX 3090参考侧，用户硬件操作，证据已归档）：** L0（ACR+ACFR）与K1（输出蒸馏）双模型TensorRT FP16部署。Jetson侧（JetPack 5.1.3/L4T R35.5.0/TRT 8.5.2.2，MAXN+锁频）：纯模型延迟P50五轮10.206–10.211/10.213–10.250ms（中位10.208/10.221，trtexec GPU均值9.571/9.599交叉验证）；图像端到端29.028/29.924FPS（mean 34.4/33.4ms）；视频稳态16.652/16.585FPS（0.555×实时）；功耗GPU电轨均/峰8.43/12.95W与8.41/12.86W（**VDD_CPU_GPU_CV合并轨=GPU上界，本板无独立GPU电轨**）、整机16.42/16.41W、稳定温度58.6/59.8°C、统一内存峰值7263/7238MB——K1与L0五类指标逐项持平，第5章零部署代价命题在边缘侧成立。冻结1000图子集AP：L0 all/small 0.272698/0.111138、K1 0.276181/0.107471（all升/small降，与10k终评方向一致）。步骤D数值一致性在G1-G4修订判据下FAIL，量化分解：5/580失配全部位于0.25阈值±0.01边界带（score∈[0.2502,0.2591]），匹配对IoU均值0.986/0.985、bulk分数漂移0.002，定性为sigmoid边界斜率×FP16量化的物理现象、引擎健全（`NUMERIC_CONSISTENCY_G_DECOMPOSITION.md`，G2/G3判据修订建议列导师追认，原始FAIL JSON保留）。RTX参考侧对跑（yolo26-trt/TRT 11.3；因其移除BuilderFlag.FP16，改用同一.pt经export(half=True)导出的fp16-native ONNX+strongly-typed，与Jetson侧fp32-ONNX+FP16旗标语义等价）：子集两侧对减四项全过（L0 all/small +0.187%/-0.085%、K1 -0.034%/+0.234%，判据≤0.5%），两侧预测清单SHA256一致；RTX FP16 b1 P50=1.6947/1.6988ms（per-forward口径）——**§11.3引擎fp16误标事项就此关闭**（服务器修复记录`SERVER_FIXES_20260924.md`：build_engine.py TRT11探测式兼容+2个缺失数据文件）。证据：`results/jetson-deploy/2026-09-24/`（results/52件含RTX8件、scripts/14个当天修复版、README_RESULTS含RTX补录节）。剩余：论文模板合并/摘要成稿收尾/导师追认清单（新增G2-G3判据修订与D步FAIL定性）；INT8按规格须另立协议且在FP16收口+导师确认之后。

**P26–P29 导师"方案B"清单补全筛选（2026-09-28起，用户授权的防盲审短板补全轮）**：**P26时段切片离线分析**（2026-09-28，`results/timeofday-slices/README.md`）——九模型全表（batch1: R0/A1/L0/K1/v10n；batch2: 原生YOLO26n/YOLO11n/v8n/C-b16/T0），属性源官方labels.zip SHA256入库、分布与冻结stats逐数一致（night 3,929/day 5,258/dawn-dusk 778/undefined 35）：夜间small AP相对退化-15~-25%（第三章问题量化）；L0 vs P2夜间small +18.76%>白天+11.21%（第四章ACFR机制）；K1蒸馏夜间增益放大（vs L0 night all +2.24%/day +0.92%）且夜间small相对退化-15.4%为全表最小（第五章机制）。描述性分析不做门，未在任何切片调参；切片内配对Bootstrap为可选后续。**P27损失族**（Inner-IoU r=0.8 / MPDIoU，协议`P27_LOSS_VARIANT_STAGE20_PROTOCOL.md`，`box_iou_variant`单点替换、结构/DFL/分类损失零改动）：M1双门未过（all -1.04%/small -0.83%）关闭；I1过晋升门（all -0.73%/small +2.90%）后续训200轮，2026-09-30终值all 0.282798/small 0.118533（vs L0-200 +0.18%/+1.69%），1000次配对Bootstrap small CI95[-0.0013,+0.0049]跨零P=0.887——**采纳门未过，损失族以受控边界负结果全收口**（`a20ec90`+`1d06f56`）；Stage20 small优势随充分训练收敛消失（P23同型现象）。**P28 SimAM**（L19后零参数插入，TRT FP16预检PASS）：all -0.70%/small -2.69%未过门关闭（`406b0ce`）；昼夜切片-2.70%/-2.78%均匀损伤、无夜间特异收益。**P29 CA/RFB**（协议`P29_CONTEXT_ATTENTION_STAGE20_PROTOCOL.md`，与SimAM同插入点构成同位置受控比较族）：C1 CoordAtt（+1,552参数）2026-09-30出数——all 0.245210（**-1.15%，差0.038pp未过-1%门**）/small 0.099947（+2.24%过线），按预注册双条件门**关闭为边界负**（`e9b47af`）；切片day small +2.89%/night small -2.53%呈昼增夜损模式。R1 RFBlock（+13,376参数，dilation 1/2/3标准卷积TRT安全构造）在训（GPU1）。EMA因官方参考实现多渠道无法获取暂缓（防实现失真质疑）；LSK按U1大核DW TRT失败先例书面关闭。统一基准L0@ep20（all 0.248070/small 0.097757）、晋升门all≥-1%且small≥+2%、200轮采纳门=Bootstrap CI95下界>0。

**H3混合候选与第五章结构搜索终局（2026-09-18 01:03收口）**：用户授权B路线分析后，H1（×0.8，GMACs -6.39%）与H2（×0.75，-7.70%）构建时测量淘汰；H3（backbone×0.75+L13收缩+neck DWConv，参数-31.56%/GMACs -9.16%双门真实通过，TRT延迟预检无加速已如实入协议）Stage20完整10k评估：**all AP -6.89%、large -13.54%、small AR -3.72%，三门判负**——多层同时减宽的耦合伤害远超ACR L28单点先例与卷积替换（U1 -4.45%）。按冻结协议，宽度收缩路线与单元替换路线一并关闭。**第五章结构搜索以完整预注册负结果收口：5个架构族（DWConv/GhostConv/位置分解/宽度收缩/混合）全部受控证据闭环，损伤定位（backbone语义通路容量）、压缩-延迟陷阱（2.2%延迟封顶+奇通道反噬）均有一手数据。A路线（负结果+机制分析章节，第六章只部署L0）为默认去向，导师追认待办。**证据：`results/lightweight-h3/`、`docs/protocols/LIGHTWEIGHT_H3_HYBRID_STAGE20_PROTOCOL.md`。YOLO11n同协议基线在GPU0进行中（ep115/200，周六~13:40出数）。

**P-N位置消融（2026-09-17 15:30收口，关键归因发现）**：仅neck（L23/L26）DWConv Stage20完整10k评估——**三条精度门全过**：all AP -0.66%（门≤2%）、small AP +1.28%（全部候选最高）、small AR -0.84%；medium -0.04%、large -0.69%近零损失。复杂度行如预判不足（参数-7.31%/GMACs -2.01%）不可单独当选，但与P-BN（all AP -4.45%）对照构成决定性归因：**单元筛选损伤几乎全部来自backbone两处替换点L5/L7（进入P4/P5的唯一跨通道混合入口），neck替换本身无害**——边界混合点假说获直接证实。B路线修正方向精确化：neck DWConv（无害）+ backbone宽度收缩（绕开混合点）的混合候选。P-B（仅backbone）20:27收口（用户两次让卡暂停/恢复，检查点精确接续）：all AP -4.24%、large -7.77%——**单独复现组合损伤约95%，归因完全坐实**；三行终版表见`results/reports/lightweight/lightweight_position_ablation_report.md`。同晚完成TRT运行时剖析（`lightweight_runtime_profile_report.md`）：被换4层仅占TRT延迟2.20%、三候选batch1全部持平或更慢（U1混合+9.62%）——**单元替换路线延迟上限≈2%双重封顶；宽度收缩获得独立延迟论证**。证据：`pb-size-val-ep20.json`、`pb_stage20_summary.json`、`results/lightweight-runtime/`。

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
  → P18A Haar小波下采样Stage20筛选（已完成；small -0.89%未过门，关闭）
  → P18B ACR非对称容量重分配Stage20筛选（已完成；small +9.81%过继续门）
  → ACR epoch40配对确认（已完成；+9.16%保持）
  → ACR/R0 200轮全训与配对Bootstrap（已完成；small +7.28%，CI严格>0）
  → P18C widen-only容量对齐对照200轮（已完成；与ACR统计不可分，重分配定位效率贡献）
  → ACR seed1/seed2复制（已完成；均过跨seed门）
  → P18三seed收口、推理速度基准与封箱（已完成，2026-09-07）
  → P19A ACFR实现/测试/协议/配置第一轮交付（已完成并提交）
  → P19A ACFR工程smoke与Stage20筛选（已完成；small +6.80%过继续门）
  → P19A ACFR epoch40确认（进行中，2026-09-09）
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

**P20进行中（2026-09-17）**：位置消融P-N（GPU1）训练至epoch20暂停+10k评估；P-B自ep6恢复（晚间，GPU0）。两者出数后出位置消融报告，与单元筛选负结果合并提交A/B裁决（A接受负结果改写第五章为机制分析；B协议修正后继续，候选方向见`results/reports/lightweight/unit_failure_analysis_and_directions.md`：①P4/P5宽度收缩——内部先例ACR L28 256→192近零代价；②GSConv/PConv保留混合的轻量单元；③MGD/LD风格定向蒸馏修复all AP）。裁决与方向修正均需用户+导师书面确认并另立预注册协议；未裁决前不启动S0全训与S1蒸馏。

2026-08-27补充：已记录拟P18频域/小波模块改进候选方案（`docs/plans/MODULE_IMPROVEMENT_PROPOSALS.md`），仅为候选记录，不改变上述停止边界；启动前置条件与纪律见该文档及`NEXT_STAGE_PLAN.md`第6节。

当前不得恢复PASCP、RDCI、P2到P3蒸馏、P2-960训练或历史失败模块；VisDrone test-dev继续封存。GT Oracle不得作为实际推理方法。P17按其协议关闭（只读审计已完成，判定性能失败），不得重跑或放宽。

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
| 2026-09-16 | P20-LIGHT-SPEC-AND-PREP | 完成 | 执行说明入库；分支`p20-lightweight`；L0教师复评差异=0；逐层成本审计；U1/U2实现+11项单测+ONNX/TRT预检+双smoke |
| 2026-09-17 | P20-U1-DW-STAGE20 | 完成/未过门 | 参数-24.87%、GMACs-8.07%过；small AP +0.34%过；**all AP -4.45%未过≤2%门**；TRT全FP16失败（L7混合精度可部署）；checkpoint保留不续训 |
| 2026-09-17 | P20-U2-GHOST-STAGE20 | 完成/未过门 | small AP +0.90%、all AP -2.60%（差0.60pp）；GMACs -3.97%预判负；TRT干净；两单元均败→第五章结构搜索负结果待A/B裁决 |
| 2026-09-17 | P20-ACR-ACFR-VISUALS | 完成 | 26张冻结权重图：五列对比×12、事件案例×12（含诚实案例）、G=4门控热图×2；全库事件统计6994/5909/5154/9543图 |
| 2026-09-17 | P17-UAVDT-CLOSURE-AUDIT | 完成/只读 | 判定性能失败非工程失败：仅败给detection-count 0.92%相对差，vs HGB CI>0显著；不重跑不放宽 |
| 2026-09-17 | P20-PN-NECK-STAGE20 | 完成/精度全过·复杂度不足 | all AP -0.66%、small AP +1.28%三精度门全过；参数-7.31%/GMACs-2.01%如预判不足；损伤归因backbone L5/L7 |
| 2026-09-17 | P20-PB-BACKBONE-STAGE20 | 完成/未过门 | all AP -4.24%（≈P-BN损伤的95%）；参数-17.57%过、GMACs-6.06%差0.94pp——归因坐实：病灶=L5/L7 |
| 2026-09-17 | P20-RUNTIME-PROFILE | 完成 | TRT逐层+三候选batch1：被换4层仅占延迟2.20%，候选全部持平/更慢——单元替换路线延迟封顶；A/B材料齐 |
| 2026-09-18 | P20-B-H1-H2-MEASUREMENT | 完成/淘汰 | ×0.8与×0.75均不过GMACs门（-6.39%/-7.70%），构建时测量淘汰未训练 |
| 2026-09-18 | P20-B-H3-HYBRID-STAGE20 | 完成/未过门 | 参数-31.56%/GMACs-9.16%双过；all AP -6.89%、large -13.54%、small AR -3.72%判负——宽度路线关闭，第五章负结果收口（用户授权，导师追认待办） |
| 2026-09-17 | P21-YOLO11N-BASELINE | 完成 | 用户08:31让卡暂停于ep153（第三次GPU0让卡，偏差记入P22协议§9）；20:17恢复，23:47链完成：all 0.251978/small 0.081505（vs L0-200：-10.74%/-30.08%）（`5dc2bdb`）；论文4.6.5锚点 |
| 2026-09-28 | P25-BASELINE-EXPANSION | 完成 | 用户授权补跑YOLOv8n/YOLOv10n同协议基线（对比表锚点扩充，协议逐字段同P21）；COCO权重双源交叉校验入库（v8n `f59b3d83…`、v10n `61b91ffc…`）；v10n 04:51入库（`045e2f5`）、v8n经用户恢复后05:45入库（`10fa64d`）；**终值：YOLOv8n all 0.255499/small 0.083514；YOLOv10n all 0.262477/small 0.089203**（vs L0-200 small均落后≥21%）；v8n让卡1次（ep69暂停，`c45d941`）偏差已记 |
| 2026-09-18 | P22-KD-PROTOCOL-FREEZE | 完成 | 新第五章主线冻结（用户/外部评议裁定）；教师/三臂/晋升门全部预注册；P20负结果转5.2节 |
| 2026-09-18 | P22-T0-TEACHER-STAGE20 | 完成/大幅过门 | 07:49协议冻结后全新启动，12:32链完成；all +16.74%、small +21.33% vs L0@ep20，双门≥+5%均超三倍余量（`c98b8f9`） |
| 2026-09-20 | P22-T0-TEACHER-EPOCH200 | 完成/早停收口 | ep172触发标准EarlyStopping(patience=50)，best=ep122；终版10k评估all 0.313650/small 0.131416——vs L0-200 **+11.11%/+12.74%**（`t0-size-val-ep200.json`实为终版best评估，`120951e`） |
| 2026-09-20 | P22-K-ARMS-STAGE20 | 完成/全部未过门 | 三臂vs L0@ep20：K1输出蒸馏 all +1.42%/small +1.73%（门1惜败差0.27pp）、K2特征蒸馏 -0.82%/-0.43%、K3联合 +1.22%/-0.04%（非叠加，K2拖平K1的small）；证据`42ee62e`/`5ddedfb`/`5ebaae1`。按§8负结果收口：无臂续训，第五章=边界分析+蒸馏负结果 |
| 2026-09-20 | P23-A-CONTROL-B16 | 完成/裁决依据 | C-b16@ep20：all 0.245575/small 0.097235（vs L0@ep20-b32：all -1.01%/small -0.53%——混同假设成立，b16代价=P22惜败量级）（`72e982a`） |
| 2026-09-20 | P23-B-MGD-M1 | 完成/未过门 | 与A步并行（用户指示，时序偏差已记§6）；small_kd_mgd实现+10单测+CUDA预检全过；@ep20 vs C-b16：all +2.09%/small +0.16%未过——MGD与前景L1同样不迁移small（机制证据）（`6e3cfa9`） |
| 2026-09-21 | P23-SAME-CALIBER-VERDICT | 完成/K1过门 | 同口径重判（vs C-b16，门值不变）：**K1 all +2.45%/small +2.28%门1 PASS（唯一）**；K2 +0.19%/+0.10%、K3 +2.25%/+0.49%未过；输出蒸馏=唯一有效通路 |
| 2026-09-21 | P23-K1-EPOCH200 | 完成 | K1从ep20续训至200轮（GPU0，期间第5次GPU0让卡ep127暂停、09-22 19:45恢复）；09-23 14:34链尾自动10k终评+提交：all 0.287017/small 0.116694（`810fc83`） |
| 2026-09-21 | P23-C-B16-EPOCH200 | 完成 | 用户批准同口径终评配对：C-b16对照续训至200轮（GPU1，期间第3次GPU1让卡ep193暂停、09-23 14:17恢复）；09-23 15:50链尾自动10k终评+提交：all 0.282324/small 0.118558（`21c818f`） |
| 2026-09-23 | P23-EP200-FINAL-EVAL | 完成/正结果(重定性) | 同口径主对照K1 vs C-b16：**all +1.66%（+0.47pp）Bootstrap CI95[+0.23,+0.71]pp P=1.000显著；small -1.57% CI跨零P=0.117无差异**；vs L0-200：all +1.67%显著/small +0.11%无差异；ep20 small优势(+2.28%)收敛消失。三模型复杂度逐项一致（340层/2,417,428参数/10.3 GFLOPs）；K1与C-b16 ONNX均506节点算子一致、无教师/适配器残留。证据`results/kd-arms/bootstrap-k1-vs-{c-b16,l0}-ep200.json`（`3a270de`）。5.4定性调整为"显著整体AP提升/零代价/small无显著增益"，变更列入导师追认 |
| 2026-09-23 | P23-SPEED-TEST | 完成/平价确认 | L0 vs K1-蒸馏L0：**TRT FP16 b1 -0.29%、b16 +0.24%（部署口径完全平价）**；PyTorch FP32 b1 +0.3%、b16 +2.1~3.3%（实例级噪声，两run复核L0稳定6.087ms）。证据`results/kd-arms/speed_{pt,trt}_*`×7文件；"零部署开销"承诺兑现。执行注：b16需dynamic-batch ONNX重导（首版静态b1），ONNX与engine缓存`weights/p23_bench/`（gitignored） |
| 2026-09-25 | P24-DEPLOY-ARCHIVE | 完成 | 第六章Jetson+RTX部署证据归档入库：`results/jetson-deploy/2026-09-24/`（52结果+14脚本+SERVER_FIXES）；对减四项全过、fp16误标关闭；PROJECT_STATUS/SESSION_HANDOFF同步P24 |
| 2026-09-28 | P27-LOSS-SCREEN-STAGE20 | 完成/一胜一负 | 损失族双臂筛选（协议`docs/protocols/P27_LOSS_VARIANT_STAGE20_PROTOCOL.md`）：**I1 Inner-IoU(r=0.8)过门**（all -0.73%/small +2.90% vs L0@ep20）当日17:46续训200轮；M1 MPDIoU双门未过（all -1.04%/small -0.83%）关闭（`571bc7a`）。接线经E2ELoss探针验证（one2many+one2one双路径生效）；M1曾于ep3让卡1次 |
| 2026-09-29 | P28-SIMAM-STAGE20 | 完成/未过门 | SimAM零参数结构臂（协议`P28_SIMAM_STAGE20_PROTOCOL.md`，L19后单点插入）：训练前工程门全过（参数delta=0/ONNX 164层/TRT FP16双模型PASS）后Stage20出数 **all 0.246339（-0.70%）/small 0.095123（-2.69%）**——small门未过关闭；同轮夜间切片-2.70%/昼-2.78%无夜间特异收益（`406b0ce`）。方案B结构族收口为受控负结果 |
| 2026-09-30 | P27-I1-EPOCH200-FINAL | 完成/未采纳 | I1 Inner-IoU 200轮终值 all 0.282798/small 0.118533（vs L0-200 +0.18%/+1.69%）；Bootstrap small CI95[-0.0013,+0.0049]跨零P=0.887、all跨零P=0.615——采纳门未过，**关闭为边界负结果**（`a20ec90`+bootstrap JSON）。Stage20 small优势随训练收敛消失，同P23现象。P27损失族全收口 |
| 2026-09-28 | P26-TIMEOFDAY-SLICES | 完成 | 九模型时段切片全表（`results/timeofday-slices/`，官方labels.zip SHA256 `7f1f9043…`与冻结stats逐数一致；night 3,929/day 5,258/dawn-dusk 778/undefined 35）：夜间small AP相对退化-15~-25%（第三章问题量化）；L0 vs P2夜间small +18.76%>白天+11.21%（第四章ACFR机制）；K1蒸馏夜间增益放大（vs L0 night all +2.24%/day +0.92%）、夜间small退化-15.4%全表最小（第五章机制）。描述性分析不做门、未在切片调参；切片内配对Bootstrap为可选后续 |
| 2026-09-29 | P29-CONTEXT-ATTENTION-FREEZE | 完成 | 导师方案B剩余两项冻结（协议`P29_CONTEXT_ATTENTION_STAGE20_PROTOCOL.md`）：C1 CoordAtt（+1,552参数）/R1 RFBlock（+13,376，dilation 1/2/3标准卷积TRT安全构造），L19后单点插入与SimAM构成同位置受控比较族；TRT FP16预检双PASS、9单测+177回归绿（`28ccb1d`）；EMA暂缓（参考实现多渠道无法获取，防实现失真质疑）、LSK按U1大核DW TRT失败先例书面关闭。冒烟两次失败根因=模板device带引号致sed未命中误上GPU0，修正后23:51经用户授权正式链启动（`175fc90`） |
| 2026-09-30 | P29-CA-STAGE20 | 完成/未过门(边界) | C1出数：all 0.245210（**-1.15%，差0.038pp未过-1%门**）/small 0.099947（+2.24%过线）——双条件晋升门all未过即关闭，**边界负结果**（`e9b47af`）；切片day small +2.89%/night small -2.53%昼增夜损模式。R1 RFB在训（GPU1，ep14/20@晨）待同门判定 |
| 2026-09-30 | P29-RFB-STAGE20 | 完成/**双门全过** | R1出数：all 0.249508（+0.58%）/small 0.103451（**+5.82%**，门+2%近三倍余量）——方案B六族筛选唯一过晋升门（`d6c6a78`）；切片day +6.34%/night +5.63%昼夜双正（背景抑制假设成立画像）。08:55按协议预授权启动200轮续训（GPU1，tmux `p29_rfb_ep200`，链尾含Bootstrap采纳门） |
| 2026-10-01 | P29-EMA-STAGE20 | 完成/未过门(双门) | E1臂（2026-09-30用户授权追加冻结，参考实现经GitHub API取回网络恢复后入库，factor=32参考默认+48参数，与参考逐位一致单测+182回归绿+TRT FP16预检PASS，工程`a7675a6`）出数：all 0.244122（**-1.59%未过-1%门**）/small 0.099664（+1.95%未过+2%门，差0.005pp极边界）→关闭为负结果（`cb970b0`）；切片day small +1.73%/night small -0.68%昼弱增夜平损。**方案B七项全部受控闭环，RFB唯一存活（ep104/200@10-01 04时，预计10-02凌晨出采纳门终判）** |
| 2026-10-02 | P29-RFB-EPOCH200-FINAL | 完成/未采纳(边界) | R1 ep200终值 all 0.283847（+0.55%）/small 0.119727（+2.71%）；Bootstrap small ΔCI95[-0.000243,+0.006586]下界距零仅0.024pp P=0.964、all跨零P=0.852——**采纳门（CI95下界严格>0）未过，关闭为边界负结果（全项目最接近显著的一次）**（`0f96b61`）。ep20→ep200：small +5.82%→+2.71%收敛过半未归零（优于I1/K1形态）、all +0.58%→+0.55%稳定；切片day small +3.99%/night small +2.09%昼夜双正保持（全项目唯一）。**P29三臂收口，方案B七项最终闭环为受控边界负；实验阶段就此收官，剩余论文模板合并/导师追认/备份** |
| 2026-10-11 | P29B-CUDA804-INTERRUPTION | 事故/待重启 | seed1训练完成（ep200全跑完，best.pt完好，在线best mAP50-95 0.28624较seed0高~1%）但链尾评估未跑：CUDA Error 804（驱动库580.178与内核模块不匹配，新CUDA进程无法初始化，老进程不受影响）。s2接力会话已清理；恢复脚本`scripts/p29b_recover_after_reboot.sh`已入库（幂等：seed1评估→seed2接力→三seed pooled bootstrap全流程）。**等用户重启机器后执行** |
| 2026-10-11 | P29B-SEED1-EP200-FINAL | 完成/单独显著 | **seed1终值 all 0.287118（+1.91% vs L0-s1）/small 0.122643（+4.23%）**；per-seed bootstrap small ΔCI95[+0.0023,+0.0081] P=1.000、all CI95[+0.0027,+0.0081] P=1.000——**双维度CI下界严格>0，seed1单独即显著**（`f2a9554`）。两seed方向一致（seed0 +2.71% P=0.964 / seed1 +4.23% P=1.000），符号一致性已满足。期间CUDA Error 804中断事故经用户授权重启+@reboot cron全自动恢复（seed1评估17:58补齐，seed2 17:36自动接力在训）；**pooled终判预计10-13午后** |
| 2026-10-13 | P29B-RFB-3S-POOLED-FINAL | 完成/采纳门PASS | **三seed合并pooled bootstrap：small Δ+0.0049 CI95[+0.0032,+0.0067] P=1.000、all Δ+0.0036 CI95[+0.0018,+0.0051] P=0.999——双维度CI95下界严格>0**；seed2终值small +5.29%（单独显著P=1.000）；三seed small Δ全正（+0.32/+0.50/+0.61pp）符号一致性满分（`35f73c5`）。**§7预注册pooled采纳门PASS——RFB增益经三seed复制确立为真效应**（seed0单seed差0.024pp的边界负因噪声假设被证实）。剩余：TRT FP16延迟平价（最后一步）→导师追认 |
| 2026-10-13 | P29B-TRT-LATENCY | 完成/非平价trade-off | TRT FP16实测（§11.3同口径，两轮+交换顺序四组稳定）：**b1边缘口径RFB快16.6-18.6%**（2.16-2.19 vs 2.63-2.66ms，第六章Jetson部署口径利好）；**b16批量口径RFB慢16.6-17.5%**（1.81 vs 1.55ms，P2高分辨率三分支空洞卷积b16下内存带宽放大）。按§7.2"延迟平价"字面不满足→**最终采纳连同"最多加一个"转导师裁决**（数据面：三seed精度显著+部署口径更快；代价面：批量口径慢17%）。**P29b全链收官**（226db4e判定+本条+speed JSON×4入库） |
| 2026-10-13 | P29B-RFB-ADOPTED | 导师采纳(正结果收官) | 用户告知导师采纳RFB：第四章增加ACR+ACFR+RFB增强行（三seed pooled small +4.2%/all +1.3%双显著）；第六章部署表增RFB行（b1快17%/b16慢17%双向如实）；§7.2延迟平价门以trade-off+部署口径更优获导师豁免。**遗留一致性事项**：(1) L0+RFB×K1蒸馏组合未训练，最终部署模型定义待导师明确；(2) 建议异地备份立即重打 |
| 2026-10-13 | P30-LOSS-REOPEN-FREEZE | 协议冻结/双臂在训 | RFB采纳后用户授权重开损失族（新基线L0+RFB）：N1=D-InterpIoU（arXiv:2507.12420官方参考逐位实现，lv/hv参考默认）、N2=PIoU2（Neural Networks 2024官方参考，Lambda=1.3），协议`P30_LOSS_REOPEN_ON_RFB_PROTOCOL.md`训练启动前冻结（`22e1669`）；基准=R1@ep20（即L0+RFB切面，all 0.249508/small 0.103451）；同P27门（all≥-1%&small≥+2%→ep200 Bootstrap CI下界>0）；ONNX图一致性验证替代TRT预检（523/523节点一致）；189回归绿含逐位单测。双臂并行在训（GPU0直启/GPU1门控接力——N2首启死于已知并发cache竞争已按标准门控重启）；N1预计10-14晨出数 |
| 2026-10-13 | LOCAL-SYNC-THESIS-PLANS | 完成/文档同步 | P29b全链证据打包拷回Windows本地（860MB，sha256 `24efde9e…`，解压至本地yolo/并留传输原件归档）；本地论文工作区`论文重写_20260927`新增03（RFB采纳更新清单）/04（第四章整合规划：RFB并入CGF第三单元写法、表4-2/4-4/4-6/4-7/4-9加行、4.4.9双向延迟如实交代），00-04五份文档镜像至`docs/paper/rewrite_20260927/`；**服务器侧待办：三模型同批TRT FP16重测+激活显存实测（GPU独占，待P30出数后）**；N2(PIoU2)为用户主动暂停（GPU1空闲） |
| 2026-10-14 | P30-STAGE20-FINAL | 完成/双臂判负收口 | **N1 D-InterpIoU all -2.16%/small -1.34%双门未过**（抵消RFB的small收益）；**N2 PIoU2 all -0.41%过/small -5.49%严重未过**（非单调聚焦降权低质量anchor=撤掉小目标学习信号，机制推断已记）。按§4两臂全败→**损失族在L0+RFB基线上二次关闭**（四族四负），不加第三变体（协议`P30`，证据`c1b20c8`/`bbe0a1d`）。P30收官，双卡空闲 |
| 2026-10-14 | P31-LAUNCH | 协议冻结/单臂在训 | L0+RFB 学生 × K1 输出蒸馏（最终部署模型定型臂，协议 `P31_KD_ON_RFB_PROTOCOL.md` 启动前冻结 `8a5dd13`）：蒸馏超参沿用 P23 K1 冻结值（small_kd_out/dis=1.0/T0 教师），唯一差异=学生结构 yaml 换为已采纳 L0+RFB；b16/nbs64/e200/seed0；链 `p31_chain.sh`（tmux `p31_k1_rfb`）训练→10k评估→时段切片→三 Bootstrap（vs C-b16 主门/vs L0+RFB 部署对照/vs K1 边际核对）→自动提交；健康窗 dis_loss 3.13→2.29；**ETA 10-16 傍晚**。同日：A2 三模型同批 TRT 重测入库（`62c83eb`，b1: ACR 2.431→L0 2.628→RFB 2.151；b16: 1.361→1.550→1.807）；SCI 投稿定位重写 V2（`43cb44e`）；数据集盘点（VisDrone 零样本零下载可立即跑） |
| 2026-10-17 | P31-FINAL | 完成/主门未过收口 | **ep200 终值 all 0.290052（项目历史最高）/small 0.116618**（链 v2 于 10-17 09:19 训练完成、10:15 自动评估+三 Bootstrap+提交 `29a06e4`）。三门判定：**主门 vs C-b16（同批）未过**——all Δ+0.77pp CI95[+0.52,+1.03]pp P=1.000 下界>0 达标，但 small 点估计 -1.64% 越过 -1% 容忍线；部署对照 vs L0+RFB（跨批无门）all +0.62pp 显著正/small -2.60% P=0.032 显著负；RFB 边际核对 vs K1（非门）all +0.30pp P=0.997 显著正/small -0.008pp≈0。**按 §3 收口：组合关闭，最终部署模型维持 L0+RFB（无 KD），导师悬置的"最终部署模型定义"就此了结**；窄负重开条款核不适用（败因为另一维容忍线、非主维度窄 CI）。机制素材入第五章/SCI：all 维度两增益近似加法叠加（0.2823→0.2834→0.2870→0.2901）；**输出蒸馏将 RFB 的 small 增益抵消回 K1 水平**（0.1197→0.1166≈K1 的 0.1167，逐位接近）。证据 `results/p31-kd-rfb/`（size-val/timeofday/三 Bootstrap 全套） |
| 2026-10-17 | P32-VISDRONE-AUTH | 协议冻结/零样本评测 | 用户确认泛化执行序，P32 VisDrone 零样本启动（GPU1）：BDD 冻结权重直接推理 VisDrone 官方 val（548 图全量），类别映射按 SCI V2 §5.C 冻结表（pedestrian+people→person、car→car、truck→truck、bus→bus、motor→motor、bicycle→bike；van/tricycle/awning-tricycle 及 rider/train/灯/标无对应丢弃，文中声明）；同评估代码（`collect_detection_predictions.py`+`evaluate_object_sizes.py`）；**不做任何选型、不接触 test-dev**；模型集=主消融链+KD 线（P2/ACR/ACFR/L0+RFB/K1/C-b16/P31）。协议 `P32_VISDRONE_ZEROSHOT_PROTOCOL.md` 启动前冻结 |
| 2026-10-17 | JETSON-AGX-REMEASURE | 完成/证据归档 | **AGX Orin Super 32GB 首轮部署重测（第六章 v2 数据源；旧 Orin NX 09-24 轮全部数字作废）**。口径与旧轮一致（MAXN_SUPER+jetson_clocks、48 预热+52×10、CUDA 事件含 H2D、5 轮取中位）。FP16/b1 纯推理：L0 4.802 / K1 4.811 / RFB 5.164 ms（trtexec 4.537/4.515/4.794 交叉验证）；图像端到端 61.3/61.1/60.1 FPS；拼接视频稳态 34.4/33.9 FPS（1.15×/1.13× 实时）；冻结 1000 图子集 AP all/small：L0 0.272786/0.111188、K1 0.276293/0.107447、RFB 0.275611/0.114115；与 RTX 3090 同批对照平台差 all ≤0.04%/small ≤0.24%，与 09-24 旧服务器值复现差 <0.03%。**RFB 引擎改 fp32-IO（keep_io_types）配方**：首测半精度图 fp16 输入端口致 CPU 转 fp16 瓶颈 9.38ms 失真，重建后 5.164ms——论文表 6-2 用此口径，halfio 全套档案保留；服务器侧同步改配方重评（差 <1e-4）。D 步 raw FAIL=top-300 槽位重排既定签名（配对 IoU≈0.986，G1-G4 全过，归导师追认）；pycuda R39 不可装经 cuda_shim 等价计时；l0 两引擎 build 记录因链重启缺失（引擎经实载验证）。证据 `results/jetson-deploy/2026-10-17/`（本地镜像 `04_Jetson部署与实测/jetson_results_20261017_AGX_Orin/`）+ `rtx-server-20261017/`（同批 RTX 复评，仅服务器副本）+ `results/`+`data/`；总说明 `README_RESULTS_20261017.md` |
| 2026-10-17 | P33-KITTI-ZEROSHOT | 完成/零样本评测 | 七 BDD 冻结权重直推 KITTI trainval 7 481 图全量（testing 7 518 未接触；Car→car/Pedestrian→person/Cyclist→bike，Van/Person_sitting/Tram/Misc/DontCare 丢弃，GT 保留 34,856/丢弃 17,009）。协议 `P33_KITTI_ZEROSHOT_PROTOCOL.md` 启动前冻结（`cceaeb1`），同评估代码同口径无门。结果：**k1-rfb all 0.251543 七模型最优**（vs p2 +2.31%；k1 +1.81%、rfb +0.04%），all/medium/large 三维最优；**small 维迁移格局与 BDD/VisDrone 不同**：acfr +6.5%/rfb +6.3% 正向而 k1-rfb −0.9%，主链 small 优势未完整迁移至 KITTI 小图幅——泛化章如实呈现，不改 BDD 采纳判定。执行偏差两处（视图脚本漏建 images/val `7e38f54`；链重启时视图检查按标签数误判跳过、手动幂等重建）不影响映射正确性，记协议 §6。证据 `results/kitti-zeroshot/`（`c1fd747`） |
| 2026-10-17 | P34-RTTS-ZEROSHOT | 完成/泛化工单收口 | **ACDC 因官方下载需注册、无合法免注册镜像（NightOwls 135GB 不可行、ExDark 非驾驶域），经用户确认改用 RTTS**（RESIDE 系雾天检测集，4 322 幅真实雾天交通图全量评测，ImageSets 不读取）。五类 car/person/bus/motorbike/bicycle 全直映零丢弃（GT 41 203 实例）。协议 `P34_RTTS_ZEROSHOT_PROTOCOL.md` 冻结（`99fcfbf`）。结果：**k1-rfb 0.271792 在 all/small/medium/large 四维全部第一——三个迁移集中唯一的全维度最优**（vs p2 all +12.59%/small +18.74%）；k1 0.270843 次之、rfb 0.255173；雾天域差未破坏 small 优势（与 KITTI small 迁移不全形成对照）。无执行偏差。证据 `results/rtts-zeroshot/`（`e4c9c4e`）。**至此 SCI V2 §5.C 泛化工单三项全部收口，服务器侧无剩余在途实验，剩余为论文侧成表** |


## 2026-10-22 公开仓库 v2 口径同步

按论文 v2 完整版(02_正文_v2_完整版_20261017)同步公开仓库:新增 ACF/CGF/RFB 全部配置与模块源码(sdf/modules/acfr.py、rfb.py)、16 个论文 v2 实验 evidence 目录(指标 JSON 白名单入库,predictions.jsonl.gz 与训练权重不入库)、yolo-system 网页系统(脱敏)、最终权重 weights/ACF-YOLO-RFB.pt。旧口径(ACR=最终方法)表述见本文档历史部分与 docs/paper/RESULTS.md,变更对照见 docs/paper/THESIS_RESULTS.md。
