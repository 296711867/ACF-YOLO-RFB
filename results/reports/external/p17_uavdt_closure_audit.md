# P17 UAVDT 外部验证收口审计（只读）

> 审计日期：2026-09-17（Asia/Shanghai）。审计人：服务器AI，按`服务器AI后续实验执行说明_优先版.md`§15执行。
> 审计对象：`.worktrees/p17-uavdt-external-validation`（分支`feat/p17-uavdt-external-validation`，工作树含未提交的在途分析工作，本次审计未做任何修改）。
> 审计问题：**Gate C失败是性能失败还是工程失败？**

## 1. 审计结论（先行）

**判定：性能失败，非工程失败。** 工程链路（数据物化、检测器训练、推理合同、一次性test访问纪律、统计机器）全部通过各自的机器检查并留有哈希证据；失败源于学习路由器在UAVDT official test上以0.92%的相对差距（small AP 0.090018 vs 0.090854）输给单一最强启发式（detection-count），未能满足预注册Gate C条件2"严格超过所有对照"的点估计要求。按协议字母（"任一条件失败……不选择单seed、不删除HGB、不改预算后重测"），关闭成立，不得重跑或放宽。

## 2. Gate C 失败的精确形态

来源：`results/uavdt-external-validation/official-test-once/gates.json`（access_id `285b817c239209f3566d`，implementation commit `4568396`）。

八项检查中**仅一项失败**：

| 检查 | 结果 |
|---|---|
| router25 small AP 相对 full640 ≥ +5% | ✅ |
| router25 all AP 降幅 ≤2% | ✅ |
| router25 small AR 严格为正且不低于对照 | ✅ |
| **router25 small AP 严格超过所有对照** | ❌（0.090018 < detection_count 0.090854，差 -0.000836 即 -0.92% 相对） |
| vs HGB 图像级 Bootstrap small AP 差值 CI 下界 >0 | ✅（[+0.004566, +0.006688]，P(Δ>0)=1.0，10,000次） |
| ≥2个预算 utility capture 超 HGB | ✅（三预算全超：0.213/0.349/0.530 vs 0.053/0.160/0.435） |
| utility-capture AUC 严格超 HGB | ✅（0.15200 vs 0.09034） |
| small AR 严格为正 | ✅ |

五个对照的25%预算small AP：detection_count **0.090854** > ScoutDet-RV 0.090018 > uncertainty_crowding 0.087555 > small_confidence_sum 0.086156 > ridge 0.085028 > HGB 0.084733。学习路由器击败了全部学习型/统计型对照（含对HGB统计显著），仅输给表现异常强的单一计数启发式。

## 3. 工程失败假设逐项排除

| 环节 | 证据 | 判定 |
|---|---|---|
| 数据物化与划分 | `composite-audit.json`：40,409公共图、30/20序列划分、0转换错误、28,631帧含ignore区（64,227个区域） | 无缺陷 |
| ignore语义 | `summary.json`：640/960分别ignore 75,830/114,591条预测——ignore机制实际生效 | 无缺陷 |
| 检测器训练 | `detector-final.json`：status=passed、failures空、csv全有限、checkpoint/results哈希在案（Stage20精确恢复+第90轮早停，恢复历史哈希化） | 无缺陷 |
| 推理工程 | `inference-preflight.json`、dry-run（`one-shot-dry-run/`、`statistics-dry-run/`）先行冻结；正式访问`access-start/complete.json`记录在案 | 无缺陷 |
| 统计机器 | 10,000次Bootstrap（seed 20260808）、策略选择哈希（`selection_sha256`）入档；dry-run与正式结果结构一致 | 无缺陷 |
| 访问纪律 | official test唯一一次访问，16,592图/20序列，与合同一致 | 无违规 |

补充证据形态学：图像级vs HGB差值CI严格为正而**序列级CI跨0**（[-0.0030, +0.0160]，P(Δ>0)=0.886）——20个序列间存在真实的表现异质性。这一形态与"跨域泛化性能波动"一致；若是管线缺陷（如映射/打分错误），图像级与序列级应同向系统性偏移。

## 4. 诚实披露（不改变判定）

1. **败给detection-count的差距无区间估计**：预注册条件2为点估计严格比较，bootstrap仅对HGB基线运行。-0.92%相对差距在序列聚集抽样的方差量级内，不能断言detection-count"显著更优"——但这正是协议选择点估计门的原因（保守方向），事后补测被"不删除HGB、不改对照后重测"条款禁止。
2. **Oracle保留率1.1306（>100%）**：25% Oracle的small AP增量超过full960全量增量，说明960增益高度集中于子集图像、预算重分配有放大效应。非缺陷，但写入论文时需解释。
3. **在途未提交工作**：工作树存在`results/resolution-routing-necessity-audit/`及相关脚本/测试（未提交），属P17收口后的后续分析线，与Gate C裁决证据无关；其去向由用户决定，本审计未读取其结论作为判据。
4. P17曾有一次训练中断重启（`detector-training-attempt0.json`、retry1 run目录），已按检查点精确恢复并哈希记录，与BDD线同类偏差处理一致。

## 5. 对论文/后续的含义

- 第六章/外部验证叙事可采用："在唯一一次UAVDT official test上，DG-ScoutRoute以统计显著性击败全部学习型对照（含HGB，P(Δ>0)=1.0），三预算效用捕获全面占优，但以0.92%相对差距未超过检测计数启发式，按预注册严格门判定为无效跨域路由器"——负结果与部分正向证据并存，如实呈现。
- 该结论**不构成**重启P17、重跑test、或修改ScoutDet-RV的理由；任何后续外部验证须按执行说明§15另立预注册协议、在第五章与Jetson完成后由导师确认。

## 6. 证据索引（全部只读引用）

- 协议：`.worktrees/p17-uavdt-external-validation/docs/protocols/UAVDT_DOMAIN_GENERAL_ROUTER_PROTOCOL.md`（§7预注册门）
- 裁决：`results/uavdt-external-validation/official-test-once/{gates,summary,bootstrap}.json`（access_id `285b817c239209f3566d`）
- 数据：`results/uavdt-external-validation/{composite-audit,dataset-audit,prepared-dataset-audit}.json`
- 检测器：`results/uavdt-external-validation/{detector-stage20,detector-final}.json`
- 状态记录：工作树`PROJECT_STATUS.md` L24、L620、L743
