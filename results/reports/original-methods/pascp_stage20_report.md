# PASCP公平Stage20结果与关闭报告

> 工作包：`P6-PASCP-STAGE20`  
> 日期：2026-08-07（Asia/Shanghai）  
> 结论：small AP与small AR均下降；PASCP关闭，不进入Stage40

## 1. 冻结协议

R0官方YOLO26n-P2与R1 P2+PASCP均使用提交`3bd378b`、BDD 70k/10k、640、batch32/nbs64、seed0、同一`yolo26n.pt`和200轮学习率计划。两组均在epoch20保存完整状态后由运行器暂停，last含optimizer/scaler/EMA且`epoch=19`。

R0用时12,494.6秒，R1用时12,565.5秒，PASCP训练时间增加0.57%。两组20行CSV全部有限；日志未发现OOM、CPU回退、NaN、Inf或fitness恢复。R0/R1在线mAP50-95为0.23383/0.23077。

## 2. 完整10k尺寸评估

两组best均以640、batch32、conf0.001、IoU0.7、maxDet300执行同一10k逐图推理和COCO风格101点尺寸评估。

| 指标 | R0官方P2 | R1 PASCP | 相对变化 | 门槛 | 结果 |
|---|---:|---:|---:|---:|---|
| all AP | 0.235178 | 0.232069 | -1.32% | 不低于-2% | 通过 |
| small AP | 0.083358 | 0.080272 | -3.70% | 至少+5% | 失败 |
| small AR | 0.257471 | 0.240951 | -6.42% | 必须正提升 | 失败 |
| medium AP | 0.267721 | 0.267753 | +0.01% | 不低于-5% | 通过 |
| large AP | 0.466453 | 0.457785 | -1.86% | 不低于-5% | 通过 |

PASCP没有造成超过保护门的整体或中大目标退化，但small AP和small AR均明显下降。当前同图矩形上下文复制没有增加可用的小目标监督，反而降低了小目标精度与召回；它不能写成有效自研方法。

## 3. 决策

按`docs/archive/protocols/PERSPECTIVE_COPY_PASTE_RESEARCH_PLAN.md`的预注册规则，PASCP在Stage20停止：

- 不恢复R1至Stage40、Stage60或200轮。
- 不搜索执行概率、复制数量、small面积阈值、垂直带、重叠阈值、羽化宽度或流水线位置。
- 不追加跨图donor、分割掩码、场景模型或seed1/seed2，因为冻结主候选尚未通过seed0阶段门。
- 代码、checkpoint、可视化和逐图预测仅作为工程与负结果证据保留；BDD有效主模型仍为官方P2-640。

PASCP可以在论文探索章节中说明“受约束框级复制没有改善BDD小目标，矩形背景污染和垂直位置代理不足可能限制真实性”，但后半句只能作为解释性假设，不能冒充已完成消融。

## 4. 证据

- R0 best/last SHA256：`ba1cf44d2fd12c91bba12cb479e7856b57e459d2d664b924fd5ac7d40bf5f34b`
- R1 best/last SHA256：`e66c27c2c1af4626e2173acf05b0e6476129448f36273c153245d04e734915d3`
- R0尺寸JSON SHA256：`3a4d6260176c125fdd48599154d792afbeb79daa50a5fa2c0b01e8e64bdf97e0`
- R1尺寸JSON SHA256：`ce904bac3b4b36e1b1460277774fa674b2a4515c6b56190d324be80b0465a078`
- R0预测SHA256：`187329ae92fd68159538342e570e9ac8d4f5472b2955a2c13cba0b8c3fc643ab`
- R1预测SHA256：`93cb9b72f4dff3d7447a4c8c60e6872cd71393ba4c2dffc13a60c0d1a0d8ded5`
- 矩阵状态SHA256：`fbe61579d91a66fcea85a8ebbbd82b9114886abe4c5df95fafd012ce4ac0f0e5`
