# RDCI公平Stage20结果与关闭报告

> 工作包：`P5-RDCI-STAGE20`  
> 日期：2026-08-07（Asia/Shanghai）  
> 结论：small AP与small AR核心门失败；RDCI关闭，不进入Stage40

## 1. 冻结协议

R0官方YOLO26n-P2与R1 P2+RDCI均使用提交`ae129c1`、BDD 70k/10k、640、batch32/nbs64、seed0、同一`yolo26n.pt`和200轮学习率计划。两组均在epoch20保存完整状态后由运行器暂停，last含optimizer/scaler/EMA且`epoch=19`。

R0用时12,538.0秒，R1用时13,179.4秒，RDCI训练时间增加5.12%。两组20行CSV全部有限；日志未发现OOM、CPU回退、NaN、Inf或fitness恢复。R0/R1在线mAP50-95为0.23383/0.23668。

## 2. 完整10k尺寸评估

两组best均以640、batch32、conf0.001、IoU0.7、maxDet300执行同一10k逐图推理和COCO风格101点尺寸评估。

| 指标 | R0官方P2 | R1 RDCI | 相对变化 | 门槛 | 结果 |
|---|---:|---:|---:|---:|---|
| all AP | 0.235178 | 0.237999 | +1.20% | 不低于-2% | 通过 |
| small AP | 0.083358 | 0.084439 | +1.30% | 至少+5% | 失败 |
| small AR | 0.257471 | 0.256035 | -0.56% | 必须正提升 | 失败 |
| medium AP | 0.267721 | 0.270701 | +1.11% | 不低于-5% | 通过 |
| large AP | 0.466453 | 0.471280 | +1.03% | 不低于-5% | 通过 |

RDCI没有造成整体或中大目标退化，但其small AP增量只有预注册门槛的约四分之一，small AR还出现负变化。该结果不能写成有效小目标改进。

## 3. 决策

按`docs/archive/protocols/DETAIL_CONTEXT_RESEARCH_PLAN.md`的预注册规则，RDCI在Stage20停止：

- 不恢复R1至Stage40、Stage60或200轮。
- 不搜索平均池化核、残差权重、分支宽度或其他注入位置。
- 不增加去除细节拆分消融或seed1/seed2，因为主候选尚未通过seed0阶段门。
- 代码、checkpoint和逐图预测仅作为工程与负结果证据保留；BDD有效主模型恢复官方P2-640。

RDCI可以在论文探索章节中说明“浅层细节/上下文残差带来约1%整体正变化，但不足以形成有效小目标方法”，不得列为最终有效原创模块。

## 4. 证据

- R0 best/last SHA256：`7d5c13012d333bd7ae139fd48da4469f603f8d68e642f9af5033ddcb571d1ecf`
- R1 best/last SHA256：`e89a64ed1cb8b70b7a4dda74df5330011ca119bd44d673386a59f87796347a76`
- R0尺寸JSON SHA256：`d77b101b63af6f411317a48e53226987adc4899ba8ad8f534033d33096d03162`
- R1尺寸JSON SHA256：`bccbf1458938de70f9546d06bbebbe4afebc0a742c2e8178e3d8936fcdb88a0c`
- R0预测SHA256：`187329ae92fd68159538342e570e9ac8d4f5472b2955a2c13cba0b8c3fc643ab`
- R1预测SHA256：`d3f8eb18b72a3aa4c445850511aec72d8765f8313425f782aeede58c9bdca1e8`
- 矩阵状态SHA256：`a39bc9d48521abe37f067f7f9515bd021cb1cf7cecd9de2e6b557ff9ff885927`

