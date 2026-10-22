# P19A ACFR Stage20 筛选报告

> 日期：2026-09-09（§1-4）；§5为2026-09-10 200轮终评补录。协议：`docs/protocols/ACFR_STAGE20_PROTOCOL.md`（训练前冻结）。
> 实施规格：`docs/plans/P19_ACFR_SERVER_IMPLEMENTATION_HANDOFF.md`。
> 判定：**Stage20通过继续门**（small AP +6.80%）；**200轮终评通过决定性证据**（small AP +5.22%，配对Bootstrap CI严格>0；见§5）。

## 1. 假设与设计

在冻结ACR（L19 P2分支64ch、L28 P5分支192ch）上，仅将L18普通`Concat`（上采样P3语义与backbone P2细节的静态拼接）替换为ACFRConcat：两路各50%通道经零初始化深度可分离残差细化；联合输入经零初始化门控产生G=4组空间logits，sigmoid后以1为中心互补增益（g_H=2W、g_L=2(1-W)）逐组加权后拼接。初始状态与Concat逐位一致（单元测试断言）。L19/L28/backbone/Detect/stride全部不变（`diff -u`证明YAML仅头注释与L18一行差异）。

成本（nc=80，thop）：+8,132参数（+0.33%）、+0.211 GMACs（+3.8%）。ONNX静态图导出通过（opset13、CPU）。

## 2. 执行

- 单元测试7项通过（恒等逐位一致、错误暴露、状态回环、优化器覆盖、CUDA AMP、单模块与宽度不变量、参数边界：新增仅`model.18.*`、无已关闭模块混入）。全量122项测试与双审计通过。
- 工程 smoke：500图3 epoch，无OOM/NaN/fitness恢复，checkpoint可保存重载；记录提交`41406dc`。
- Stage20（用户明确授权后启动）：BDD 70k/10k、640、200轮计划epoch20可恢复暂停、batch32/nbs64、seed0、GPU0，20/20完成，checkpoint完整。
- epoch40恢复事故（2026-09-09；本条为同日晚间按机器证据对早间记录的更正）：05:33首次恢复启动，05:38因约5分钟无日志输出被误判为"无声崩溃"（原记录疑原生段错误或瞬态CUDA中止；GPU0峰值23.4GB贴近上限亦有嫌疑），据此校验checkpoint（926张量与EMA一致）后启动第二次恢复；第二次尝试05:40因CUDA OOM退出——OOM错误信息显示首次进程（pid 1054349）仍持有22.62GB显存，且该进程自05:33起CPU连续计算，证明其从未终止。当前训练即首次恢复进程，epoch21起连续运行越过误判点。全过程三次记录见run目录`resume_history.json`。非协议参数偏差。

## 3. Stage20结果（对照为冻结ACR Stage20，`results/acr-stage20/summary.json`）

| 指标 | ACR@20 | ACFR@20 | 相对 |
|---|---|---|---|
| all AP | 0.246022 | 0.248070 | +0.83% |
| small AP | 0.091532 | 0.097757 | **+6.80%** |
| small AR | 0.278957 | 0.279380 | +0.15% |
| medium AP | 0.280080 | 0.282117 | +0.73% |
| large AP | 0.468930 | 0.471443 | +0.54% |

预注册继续门（small ≥+3.0% 且 all ≥-1.0% 且 AR ≥0）三项全过。在线曲线全程微幅领先ACR（+0.5~1%），与ACR自身过门时"在线小、small大"的模式一致。相对R0同期累计small AP +17.3%（ACR单独+9.8%），20轮尺度上两模块呈叠加效应。

## 4. 状态与下一步

- ~~epoch40确认进行中~~（已于2026-09-09被supervisor越界跳过，偏差见协议§7与§2更正）。
- 机器证据：`results/acfr-stage20/`（smoke_record、summary.json、a1-size-val-ep20.json、full200_summary.json、bootstrap-acfr-vs-acr-ep200.json；逐图预测JSONL被Git忽略保留本地，SHA256记录于各summary）。

## 5. 200轮终评（2026-09-10，决定性证据）

200/200轮于09-10 18:34完成（best.pt为epoch193，SHA256 `0e977cea…`）。按规格§12.1/§12.3执行10k尺寸评估（best.pt、conf 0.001、IoU 0.7、batch32、640，与ACR-200评估同参）与1,000次同图配对Bootstrap（seed 260908，baseline为ACR `r1-size-val-ep200.predictions.jsonl`）。

| 指标 | ACR-200 | ACFR-200 | 相对 |
|---|---|---|---|
| all AP | 0.280471 | 0.282295 | +0.65% |
| **small AP** | 0.110785 | **0.116565** | **+5.22%** |
| small AR | 0.308011 | 0.312877 | +1.58% |
| medium AP | 0.318594 | 0.318265 | -0.10% |
| large AP | 0.526360 | 0.521342 | -0.95% |

配对Bootstrap（1,000次，验证图像抽样不确定性）：

| 口径 | Δ均值 | 95% CI | 判定 |
|---|---|---|---|
| small AP | +0.005984 | **[+0.003070, +0.008839]** | **严格>0，支持small AP增益** |
| all AP | +0.002061 | [-0.000556, +0.005218] | 跨0；all AP写平价/不退化，不写显著提升 |

解读：

- **small AP增益保持**：+6.80%(20轮)→+5.22%(200轮)，衰减轨迹与ACR自身（+9.81%→+7.28%）同形态，终点远高于任何合理门。相对R0-200累计+12.88%，相对官方P2全训+14.86%。
- 在线全尺寸领先曾从+1.4%收窄至+0.2%（ep141-149），但未传导至small AP——再次印证本项目"在线全尺寸mAP是small增益的悲观代理"的模式；该收窄过程已如实记录于`full200_summary.json`。
- medium/large的-0.10%/-0.95%为如实记录的代价，量级与ACR当时medium -1.98%相当。
- 进入论文贡献列表前的剩余步骤（规格§12.3）：组件消融（门控vs细化）、seed1/seed2复制、batch1 PyTorch+TensorRT FP16成本（含ACR同口径补测）、导师追认（含epoch40跳过偏差）。**每项训练均需用户明确授权后启动。**

本报告至此覆盖筛选与终评两级证据；消融、seed与成本结果出来后另行补节或另建报告。

## 6. 组件消融（2026-09-15，描述性归因，双臂完成）

按`docs/protocols/ACFR_ABLATION_PROTOCOL.md`执行：nr（gate-only，去细化，+5,316参数）与ng（refine-only，去门控，+2,816参数）各200轮全训，完整10k尺寸评估与ACFR终评同参（best.pt、conf 0.001、IoU 0.7、batch32、640）。对照为冻结ACR-200（all 0.280471 / small 0.110785）。

| 口径 | ACR-200 | nr（仅门控） | ng（仅细化） | ACFR完整（seed0） |
|---|---|---|---|---|
| all AP | 0.280471 | 0.281337（+0.31%） | 0.278683（-0.64%） | 0.282295（+0.65%） |
| **small AP** | 0.110785 | **0.114667（+3.50%）** | **0.112711（+1.74%）** | 0.116565（+5.22%） |
| small AR | 0.308011 | 0.304041 | 0.306004 | 0.312877 |

描述性配对Bootstrap（对ACR-200，1,000次，seed 260915）：

| 口径 | Δ均值 | 95% CI | 判定 |
|---|---|---|---|
| nr small AP | +0.003883 | **[+0.001177, +0.006674]** | **严格>0** |
| nr all AP | +0.000865 | [-0.001726, +0.003595] | 跨0，平价 |
| ng small AP | +0.001926 | [-0.000614, +0.004699] | 跨0，不显著 |
| ng all AP | -0.001788 | [-0.004408, +0.005144] | 跨0，平价 |

判定规则落表（协议§4）：

- **细化必要门（nr增益 < +2.0%）未触发**：nr +3.50%，去掉细化后仍保留完整ACFR增益的约65%。
- **门控必要门（ng增益 < +2.6%）触发**：ng +1.74%，去掉门控后增益不足完整ACFR增益的一半。
- 非单组件主导情形（其条件ng≥+2.6%且nr<+2.0%与观测相反）；非耦合依赖情形（nr≥+2.0%）。

归因：**small AP增益主要来自互补分组门控（必要，且单独统计显著）；部分残差细化单独不显著（CI跨0）、非必要，但与门控近似线性叠加（+3.50%与+1.74%之和≈完整版+5.22%）**。论文归因表述向门控主导调整，与ACR"加宽主导、重分配效率"的诚实归因模式一致。两变体all AP均平价（CI跨0）。

执行偏差（协议§7补记）：09-14 09:51服务器重启导致双臂中断（nr ep127、ng ep100），当日从检查点精确恢复；09-15 ~08:30 ng再次被不明外部原因中断（无重启/OOM记录），10:03恢复。所有恢复均经`resume_experiment.py` preflight（检查点/CSV一致性+EMA逐张量匹配），事件链见各run目录`resume_history.json`。

机器证据：`results/acfr-stage20/ablation_summary.json`（判定表）、`nr-size-val-ep200.json`、`ng-size-val-ep200.json`（含predictions.jsonl）、`bootstrap-nr-vs-acr-ep200.json`、`bootstrap-ng-vs-acr-ep200.json`。

## 7. TensorRT FP16 成本（2026-09-15，独立环境，规格§12.3收口）

经用户授权新建独立 conda 环境 `yolo26-trt`（TensorRT 11.3.0.99，冻结`yolo26-sdf`环境零改动；环境审计见`environment/tensorrt_env_audit.md`）。ACR r1与ACFR a1的stage20 best权重以`half=True`导出ONNX（opset 17，动态batch），TRT 11以strongly-typed network构建FP16 engine（传统`BuilderFlag.FP16`已在TRT 11移除）。计时口径：**纯engine前向**（输入H2D含在计时区，CUDA事件），图像列表/warmup/timed/repeat与PyTorch FP32基准一致（`bench_images_100.txt`、GPU0、repeat 3）。绝对值与PyTorch端到端口径不可比（后者含预处理与Python开销），**只比较同口径内ACFR对ACR的相对差**；两engine输出经sanity检查（`(1,300,6)`检测张量，数值范围正常）。

| 口径 | ACR ms/图 | ACFR ms/图 | ACFR相对差 |
|---|---|---|---|
| TRT FP16 batch1 | 2.1603 | 2.1500 | **-0.48%**（run间波动约±5%，平价） |
| TRT FP16 batch16 | 1.9910 | 1.8214 | **-8.52%**（run间稳定，ACFR更快） |
| PyTorch FP32 batch1（对照） | 5.909 | 5.958 | +0.83% |
| PyTorch FP32 batch16（对照） | 5.933 | 5.923 | -0.17% |

结论：**TensorRT FP16部署口径下ACFR对ACR成本平价至更优**——batch1延迟平价（噪声级），batch16吞吐口径快8.52%（三run稳定：1.81/1.82/1.84 vs 1.96/1.96/2.05 ms）。论文成本表述可写"PyTorch FP32与TensorRT FP16下均零额外延迟，TRT FP16 batch16下更快"。证据：`results/acfr-stage20/inference_trt_fp16_gpu0_batch{1,16}.json`、脚本`scripts/benchmark_tensorrt_fp16.py`。

剩余（规格§12.3）：导师追认创新口径与ep40跳过偏差。
