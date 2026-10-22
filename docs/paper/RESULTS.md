> **口径声明(2026-10-22)**:本文件 §1.1–§1.10 为 ACR 阶段的冻结审计记录,其"最终方法"表述已被论文 v2 口径取代:
> 最终模型 = **ACF-YOLO-RFB**(ACR + CGF 互补门控融合 + RFB 感受野增强),数字与证据对照见 [THESIS_RESULTS.md](THESIS_RESULTS.md);知识蒸馏为论文第五章方法(ACF-YOLO-KD),非负结果,只是未选入最终部署。§4 负结果边界中与论文 v2 冲突的条目以 THESIS_RESULTS.md 为准。

# YOLO26-SDF正式结果

> 更新：2026-09-06（Asia/Shanghai）
> 规则：只收录有完整训练、独立评估和哈希证据的结果；筛选实验与最终结果分开。AP/AR默认采用项目COCO尺寸口径（101点、IoU 0.50:0.95、maxDet300）。
>
> **版本状态说明（2026-09-07）**：第1-6节（除1.10外）为P11/P15时期冻结并审计的实证研究证据（分辨率路由主线）。第1.10节为2026-08-27至09-07新增的P18线（ACR自研结构），全部数字与`results/acr-stage20/`机器证据一一对应，含三seed终表；该线为用户授权启动的筛选级证据升级为完整证据链，**导师追认创新口径前暂不进入学校论文正文主表**。P18实验线已封箱，无在训任务。

## 1. BDD100K正式结果

BDD100K使用70,000 train / 10,000 val、640、batch64、seed0、200 epoch。

| 模型 | 参数量 | GFLOPs | all AP | small AP | medium AP | large AP | small AR | 训练时间 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLO26n | 2,507,700 | 5.8 | 0.255413 | 0.088772 | 0.290918 | 0.508208 | 0.237142 | 17.08 h |
| 官方YOLO26n-P2 | 2,518,352 | 7.5 | 0.268221 | 0.101480 | 0.321219 | 0.519476 | 0.293165 | 35.03 h |
| P2+浅层通道注意力 | 2,522,512 | 约7.5 | 0.268551 | 0.101622 | 0.305310 | 0.517338 | 0.288169 | 35.26 h |

官方P2相对原生YOLO26n：all AP +5.01%、small AP +14.32%、small AR +23.62%，作为BDD主模型保留。浅层通道注意力small AP仅+0.14%、small AR -1.70%、medium AP -4.95%，未通过预注册门并淘汰。

BDD证据：

- 原生报告：`results/reports/baselines/phase1_baseline_640_report.md`；best SHA256 `0fb7a0716d3ac8b646dae9aa3477b88380178a9bcf4a8094dd970ef96cb59941`。
- 官方P2报告：`results/reports/baselines/phase1_official_p2_640_report.md`；best SHA256 `f1a8ad980cdaf949a12ac98514b3ccc583e1026fe4b09aa664e8d2589bc2826c`。
- 通道注意力负结果：`results/reports/bdd-modules/p2_channel_attention_final_report.md`；best SHA256 `53e3c7d4bcf641175ead201a5999c3b27d7656ff2b1e9c717952fb4f9a58f5a3`。

P2-960 epoch39仅作精度-成本趋势，不进入同预算主表：all AP 0.28461、small AP 0.12123，约16.97 GFLOPs、22.68分钟/epoch，完整200轮估计75.6小时；详见`results/reports/bdd-modules/p2_resolution_960_epoch39_report.md`。

### 1.1 BDD配对图像Bootstrap

基于同一10,000张val图像执行1,000次配对非参数Bootstrap。all AP配对增量均值为`+0.01293`，95%百分位区间`[+0.00998,+0.01563]`；small AP增量均值为`+0.01297`，区间`[+0.00992,+0.01621]`，两者1,000/1,000次为正。该结果衡量验证样本不确定性，不替代训练多seed。完整报告：`results/reports/analysis/bdd_paired_bootstrap_report.md`；机器结果：`results/bdd-paired-bootstrap/summary.json`。

### 1.2 BDD冻结预测误差分析

在置信度0.25、同类一对一IoU 0.50匹配下，完整10k val中得到6,010个P2修复small漏检、52,563个两模型共同small漏检和12,948个P2新增误检事件。原生模型未匹配的58,573个small GT中有10.26%在该固定阈值下被P2修复；共同失败仍占主体，且按面积排序的前三个GT仅2.30–3.98像素²。新增误检前三名为高置信度bus/truck框，表明P2的召回收益伴随额外未匹配预测。上述为阈值事件分析，不替代正式AP/AR。

每类候选均先完整落盘再按冻结排序选择前三个不同图像，没有看图后替换。报告：`results/reports/analysis/bdd_error_analysis_report.md`；机器清单、输入/输出哈希及三组PNG/PDF：`results/bdd-error-analysis/`。

### 1.3 P2到P3蒸馏Stage20

四组使用同一原生YOLO26n学生初始化、640、batch16/nbs64、seed0和200轮调度，在epoch20保存完整状态后暂停。

| 组别 | all AP | small AP | small AR | medium AP | large AP | 决策 |
|---|---:|---:|---:|---:|---:|---|
| D0原生control | 0.226392 | 0.073483 | 0.205715 | 0.252215 | 0.482080 | control |
| D1公共层KD | 0.217824 | 0.068147 | 0.194927 | 0.242745 | 0.477988 | 停止 |
| D2跨层目标+small掩码 | 0.226810 | 0.074534 | 0.207550 | 0.252766 | 0.481962 | 停止 |
| D3 D2+双分支共识 | 0.226618 | 0.074529 | 0.206599 | 0.251753 | 0.484899 | 停止 |

D2/D3 small AP相对D0仅+1.43%/+1.42%，低于预注册+5%门；D1明显退化。三候选均不进入Stage40/200轮。报告：`results/reports/original-methods/p2p3_stage20_report.md`；机器结果与哈希：`results/p2p3-stage20/`。

### 1.4 RDCI Stage20

官方P2 control与P2+RDCI使用BDD 70k/10k、640、batch32/nbs64、seed0和相同200轮计划，在epoch20保存完整状态后暂停。

| 组别 | all AP | small AP | small AR | medium AP | large AP | 决策 |
|---|---:|---:|---:|---:|---:|---|
| R0官方P2 | 0.235178 | 0.083358 | 0.257471 | 0.267721 | 0.466453 | control |
| R1 RDCI | 0.237999 | 0.084439 | 0.256035 | 0.270701 | 0.471280 | 停止 |

RDCI相对R0的all AP提高1.20%，但small AP仅提高1.30%，small AR下降0.56%；两项核心门失败，不进入Stage40/200轮。报告：`results/reports/original-methods/rdci_stage20_report.md`；机器结果与哈希：`results/rdci-stage20/`。

### 1.5 PASCP Stage20

官方P2 control与P2+PASCP使用BDD 70k/10k、640、batch32/nbs64、seed0和相同200轮计划，在epoch20保存完整状态后暂停。

| 组别 | all AP | small AP | small AR | medium AP | large AP | 决策 |
|---|---:|---:|---:|---:|---:|---|
| R0官方P2 | 0.235178 | 0.083358 | 0.257471 | 0.267721 | 0.466453 | control |
| R1 PASCP | 0.232069 | 0.080272 | 0.240951 | 0.267753 | 0.457785 | 停止 |

PASCP相对R0的all AP下降1.32%，small AP下降3.70%，small AR下降6.42%；两项核心门失败，不进入Stage40/200轮。报告：`results/reports/original-methods/pascp_stage20_report.md`；机器结果与哈希：`results/pascp-stage20/`。

### 1.6 PGSR开发集可行性与分辨率路由上限

以下只使用BDD train哈希固定500图与官方P2 e200冻结权重，不属于完整val正式结果。

| 方法 | all AP | small AP | small AR | 相对B0 all | 相对B0 small AP | 相对B0 small AR | 决策 |
|---|---:|---:|---:|---:|---:|---:|---|
| B0全图640 | 0.283551 | 0.115444 | 0.297049 | - | - | - | 对照 |
| B1同权重全图960 | 0.307551 | 0.132570 | 0.331863 | +8.46% | +14.83% | +11.72% | 分辨率信号通过 |
| B2标准SAHI | 0.159292 | 0.050379 | 0.118423 | -43.82% | -56.36% | -60.13% | 裁块关闭 |
| B3 GT Oracle Top-2裁块 | 0.239944 | 0.064897 | 0.164752 | -15.38% | -43.78% | -44.54% | 裁块关闭 |
| 25% GT全图960路由 | 0.287084 | 0.128345 | 0.332443 | +1.25% | +11.18% | +11.92% | 通过上限门 |
| 50% GT全图960路由 | 0.285331 | 0.129672 | 0.342025 | +0.63% | +12.32% | +15.14% | 成本曲线 |

裁块即使覆盖99.71% small GT中心仍退化，故不再训练裁块路由。25%全图分辨率路由只通过GT上限门，尚非有效自研模块；下一步训练不读取GT的P2轻量路由头。完整协议、停止门与哈希见`results/reports/original-methods/pgsr_feasibility_report.md`，机器结果见`results/pgsr-feasibility/`。

### 1.7 透视分带学习路由开发结果

以下仍为与训练5,000图互斥的BDD train 500图开发结果，不进入正式主表。

| 策略 | all AP | small AP | small AR | 相对B0 small AP | 相对B0 small AR |
|---|---:|---:|---:|---:|---:|
| B0全640 | 0.283551 | 0.115444 | 0.297049 | - | - |
| 哈希随机25% | 0.280176 | 0.118992 | 0.296685 | +3.07% | -0.12% |
| 粗检small置信度和25% | 0.284312 | 0.122799 | 0.315192 | +6.37% | +6.11% |
| 学习路由seed0 | 0.283942 | 0.123838 | 0.326433 | +7.27% | +9.89% |
| 学习路由seed1 | 0.283716 | 0.123940 | 0.322995 | +7.36% | +8.73% |
| 学习路由seed2 | 0.284871 | 0.126077 | 0.324001 | +9.21% | +9.07% |

三seed均通过开发门；主checkpoint固定seed0而非最优seed2。该表只记录设计冻结前的开发证据，正式结果与严格判定见下一节。工程报告：`results/reports/original-methods/pgsr_router_prep_report.md`；正式协议：`docs/protocols/PERSPECTIVE_ROUTER_FINAL_PROTOCOL.md`。

### 1.8 透视分带路由正式10k结果

| 策略 | all AP | small AP | small AR | medium AP | large AP |
|---|---:|---:|---:|---:|---:|
| B0全640 | 0.268234 | 0.101503 | 0.293224 | 0.321251 | 0.519612 |
| B1全960 | 0.289446 | 0.129221 | 0.340283 | 0.331795 | 0.487788 |
| 哈希随机25% | 0.274552 | 0.106876 | 0.304020 | 0.312412 | 0.520432 |
| 粗检启发式25% | 0.280705 | 0.116185 | 0.315812 | 0.325332 | 0.516765 |
| 学习路由25% | 0.279781 | 0.115461 | 0.318907 | 0.334305 | 0.513831 |
| GT Oracle 25% | 0.281979 | 0.118889 | 0.335218 | 0.325549 | 0.512027 |

学习路由相对B0的all/small AP和small AR提高4.31%/13.75%/8.76%，基础有效性成立；但small AP略低于等预算粗检启发式，未通过预注册严格比较门。结果作为正向候选与负向必要性消融保留，不进入最终有效方法主张。报告：`results/reports/original-methods/perspective_router_final_report.md`。

正式判定后使用冻结预测补做1,000次同图像配对Bootstrap。学习路由相对B0的all AP差值95%区间为`[+0.00942,+0.01338]`，small AP为`[+0.01102,+0.01693]`，均1,000/1,000次为正；相对粗检启发式的all AP区间为`[-0.00666,+0.00110]`，small AP为`[-0.00282,+0.00170]`，均跨0。因此正收益获得统计支持，但现有样本不能区分学习路由与简单启发式的AP；这不改变预注册点估计门失败。完整报告：`results/reports/analysis/router_paired_bootstrap_report.md`。

### 1.9 集成运行时

在RTX 3090、FP32、显式batch16下，使用正式val清单前500图对五种完整流水线分别预热并重复3次。计时覆盖图像路径输入、推理、路由、125图960复检、CPU Results物化和结果替换，不含模型加载与预热。

| 策略 | 总时间（秒） | 吞吐（图/秒） | 峰值allocated（MiB） | 相对全960 |
|---|---:|---:|---:|---:|
| 全640 | 2.8839±0.1675 | 173.757±9.765 | 451.69 | -20.19% |
| 全960 | 3.6136±0.0868 | 138.419±3.291 | 913.93 | 0.00% |
| 随机25% | 3.8250±0.0669 | 130.746±2.264 | 912.20 | +5.85% |
| 粗检启发式25% | 3.7723±0.0036 | 132.544±0.126 | 912.20 | +4.39% |
| 学习路由25% | 3.7802±0.0114 | 132.268±0.399 | 912.22 | +4.61% |

学习路由选图集合与冻结正式分数一致，但总时间比全960慢4.61%，三个25%策略均未形成速度优势。当前方法只能作为固定复检预算下的精度研究，不能宣称加速或更优精度-速度前沿。报告：`results/reports/analysis/integrated_router_runtime_report.md`；机器结果：`results/integrated-router-runtime/benchmark.json`。

### 1.10 P18：非对称容量重分配（ACR）与容量对照

#### 1.10.1 设计与机制证据

ACR将官方P2的L19 P2检测分支32→64通道（与P3容量对齐）、L28 P5头分支256→192通道，backbone逐层不变。参数2,490,624（较官方P2净减6.45%），GMACs 5.579（thop同工具口径，+17.1%）。机制链：§1.2误差分析发现容量倒挂（small目标由最薄的32通道分支负责）；Haar小波无损下采样对照（P18A）small AP -0.89%反证"信息保存非瓶颈"，表示容量才是。

#### 1.10.2 200轮主结果（batch32/nbs64、seed0、同协议配对）

| 模型 | 参数 | GMACs | all AP | small AP | small AR | medium AP | large AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| R0 官方P2（对照） | 2,662,400 | 4.765 | 0.269294 | 0.103266 | 0.287891 | 0.325026 | 0.519349 |
| **ACR** | **2,490,624** | 5.579 | **0.280471** | **0.110785** | **0.308011** | 0.318594 | 0.526360 |
| widen-only对照 | 2,704,064 | 5.665 | 0.280530 | 0.109599 | 0.304433 | 0.317196 | 0.528140 |

- ACR相对R0：all AP +4.15%、small AP **+7.28%**、small AR +6.99%；1,000次配对Bootstrap差值95%区间all `[+0.00848,+0.01338]`、small `[+0.00470,+0.01062]`，均严格>0。
- 相对官方P2全训主模型（b64参考口径，§1主表0.268221/0.101480）：all AP +4.57%、small AP **+9.17%**；Bootstrap区间同样严格>0。
- widen-only（仅L19加宽、L28不减配）：与ACR精度统计不可分（配对Bootstrap差值CI均跨0）。诚实结论：P2加宽是small增益主导因素；非对称重分配以三者最低参数取得同等精度，定位为效率贡献。
- 增益稳定性：Stage20 +9.81% → epoch40 +9.16% → 200轮 +7.28%，无长训衰减。
- 推理速度（RTX 3090独占、FP32、batch16、112图、3次重复）：ACR 5.844ms/图 vs 官方P2 6.103ms/图，**-4.24%**（GMACs增加但实测更快，P5减配降低访存）。

#### 1.10.3 跨seed稳健性

| seed | all AP | small AP | small AP相对R0 | 判定 |
|---|---:|---:|---:|---|
| seed0 | 0.280471 | 0.110785 | +7.28% | 过门 |
| seed1 | 0.279051 | 0.109391 | +5.93% | 过门 |
| seed2 | 0.279852 | 0.108682 | +5.24% | 过门 |
| **均值 ± 样本标准差** | **0.279791 ± 0.000712** | **0.109619 ± 0.001070** | **+6.15% ± 1.04%** | 三seed全过 |

预注册门：small AP相对R0为正且≥+3.6%；三seed全部通过，small AP变异系数0.98%，按协议无剔除。small AR均值±标准差为0.309059 ± 0.003473。机器证据：`results/acr-stage20/three_seed_final_summary.json`（含seed2 checkpoint哈希）。

证据：协议`docs/protocols/ACR_STAGE20_PROTOCOL.md`（及WIDEN/SEED1/SEED2）；报告`results/reports/bdd-modules/acr_stage20_report.md`§1-9；机器证据`results/acr-stage20/`（full200_summary.json、widen_control_summary.json、seed1_final_summary.json、四份Bootstrap、inference_benchmark_gpu0.json）。P18A Haar负结果：`results/reports/bdd-modules/haar_downsample_stage20_report.md`。

## 2. VisDrone模型选择

VisDrone train/val为6,471/548图，所有模型均从COCO `yolo26n.pt`初始化，640、batch16/nbs64、seed0、相同100-epoch调度；下表为epoch20阶段门。

| 候选 | all AP | small AP | small AR | medium AP | large AP | 官方兼容AP(%) | 决策 |
|---|---:|---:|---:|---:|---:|---:|---|
| 原生YOLO26n control | 0.144007 | 0.073429 | 0.193117 | 0.214769 | 0.363359 | 15.2394 | 保留并续训 |
| 官方P2 | 0.141884 | 0.075710 | 0.179755 | 0.209470 | 0.313153 | 15.5235 | 淘汰 |
| 原生+`cls_pw=0.25` | 0.148289 | 0.075208 | 0.184068 | 0.224907 | 0.391637 | 15.6789 | 淘汰 |

P2 small AP仅+3.11%、small AR -6.92%、large AP -13.82%；类别权重small AP仅+2.42%、small AR -4.69%。两者均未通过各自预注册门，不续训100，不进入test-dev。

## 3. VisDrone原生e100最终结果

seed0正常完成100/100，最佳epoch72。最终`best.pt` SHA256为`8c9deeca9eff76a5461dd4567d5544c527b9d57546140a08b0dcba979176583c`。

| split | all AP | small AP | medium AP | large AP | small AR | 官方兼容AP(%) | 官方AR500(%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| val | 0.180351 | 0.098479 | 0.267417 | 0.427754 | 0.221136 | 18.9947 | 35.0259 |
| test-dev | 0.143200 | 0.064963 | 0.220325 | 0.329504 | 0.186573 | 18.3414 | 35.2369 |

test-dev在模型、权重和推理配置冻结后只访问一次，现永久封存。内部尺寸口径test-dev small AP较val低34.03%，是当前最明显的泛化短板；官方兼容AP仅低3.44%，差异来自ignored-region处理与指标聚合，禁止混用。

冻结seed0模型在val上将输入从640提高到960时，all/small AP和small AR从`0.180351/0.098479/0.221136`提高到`0.217306/0.140552/0.286036`。固定25% GT Oracle达到`0.201256/0.125729/0.261011`，保留full960的64.77% small AP增量。该结果是新路由假设的上限开发证据，不是可部署方法或未见test-dev验证。

完整报告：`results/reports/visdrone/visdrone_native_e100_final_report.md`。原始结果JSON位于`results/visdrone-native-e100/`。

## 4. 负结果边界

- SDD：all/small AP相对同源P2约-49.78%/-61.48%，淘汰。
- DWF及nearest降级：均未通过整体与small门，淘汰。
- SA-NWD三组：small局部收益不足以覆盖all AP退化，淘汰。
- P2浅层通道注意力：短子集收益未在完整BDD复现，淘汰。
- P2到P3小目标蒸馏：D2/D3 Stage20 small AP仅+1.43%/+1.42%，低于预注册+5%门；D1退化，路线关闭。
- RDCI：Stage20 all AP +1.20%，但small AP仅+1.30%、small AR -0.56%，路线关闭。
- PASCP：Stage20 all AP -1.32%、small AP -3.70%、small AR -6.42%，路线关闭。
- Haar小波无损下采样（P18A）：Stage20 small AP -0.89%（all AP平价、small AR +0.58%），低于关闭线；机制结论"信息保存非瓶颈"，作为ACR容量假设的反证消融保留。
- VisDrone P2与类别权重：Stage20核心small AP/AR门失败，淘汰。
- 检测前效用排序路由：BDD train/dev开发筛选中，seed0 utility capture为0.36375，低于同成本固定岭回归图像统计基线0.41027（-11.34%）；按预注册门停止，未访问BDD正式val。
- ScoutDet-RV：VisDrone train序列隔离holdout上三seedutility capture均超过HistGradientBoosting，三seed下置信界为0.83690，但相对强树基线0.80975仅+3.35%，低于预注册+5%门；在val前停止，不作为有效方法。

## 5. VisDrone三seed稳健性

seed0/1/2均按冻结原生YOLO26n e100协议完成，只使用train/val；test-dev未再次访问。

| val指标 | seed0 | seed1 | seed2 | 均值 ± 样本标准差 |
|---|---:|---:|---:|---:|
| all AP50-95 | 0.180351 | 0.178113 | 0.176816 | 0.178427 ± 0.001788 |
| small AP50-95 | 0.098479 | 0.098938 | 0.097610 | 0.098342 ± 0.000674 |
| small AR50-95 | 0.221136 | 0.221646 | 0.218611 | 0.220464 ± 0.001625 |
| medium AP50-95 | 0.267417 | 0.262276 | 0.263037 | 0.264243 ± 0.002775 |
| large AP50-95 | 0.427754 | 0.432838 | 0.406471 | 0.422354 ± 0.013988 |
| 官方兼容AP50-95(%) | 18.9947 | 18.8792 | 18.8919 | 18.9219 ± 0.0633 |

small AP/AR变异系数仅0.69%/0.74%，官方AP变异系数0.33%，核心结论稳定。完整报告：`results/reports/visdrone/visdrone_multiseed_report.md`；机器可读统计：`results/visdrone-multiseed/summary.json`。

## 6. 论文图表与章节初稿

`results/paper_figures/`包含5组汇总PNG/PDF及`manifest.json`，分别覆盖BDD尺寸AP、VisDrone三seed训练曲线、尺寸误差条、val/test尺寸差距和分类别泛化差距；`results/bdd-error-analysis/`另含3组BDD定性误差PNG/PDF和完整候选清单。图表只读取冻结JSON/CSV/JSONL，可分别由`scripts/generate_paper_figures.py`和`scripts/analyze_bdd_errors.py`复现。无编号完整正文见`docs/paper/THESIS_MANUSCRIPT_DRAFT.md`；学校模板接入清单、分项结果稿和方法图注已于2026-09-07归档至`docs/paper/archive/`用于复查（数字以本文为准，均不含P18 ACR）。`scripts/audit_paper_claims.py`已覆盖整合稿的核心数值、引用与图件。
