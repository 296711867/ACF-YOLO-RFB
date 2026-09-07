# RDCI工程验收与Stage20准备报告

> 工作包：`P5-RDCI-PREP`  
> 日期：2026-08-06（Asia/Shanghai）  
> 结论：工程门通过；只允许启动官方P2/RDCI公平Stage20

## 1. 文献与方法边界

DCAS论文及作者GPL-3.0仓库在BDD100K上公开报告YOLOv8l mAP50-95从34.7%提高到36.8%。本项目没有复制其DSPPF、硬编码数据扫描或P2-P4整网结构，只引用最低层细节注入动机。自研RDCI在YOLO26官方P2融合位置对P1做细节/上下文拆分，以零初始化残差注入P2，再保留原top-down concat。

## 2. 工程结果

- 官方P2/RDCI参数：2,518,352/2,527,760，增量9,408（0.37%）。
- 官方P2/RDCI GFLOPs：7.540/8.026，增量6.44%。
- stride均为`[4,8,16,32]`；加载相同官方状态后初始最终预测逐元素相等。
- COCO`yolo26n.pt`迁移360/910项，迁移数量不低于官方P2。
- 完整回归53项：48通过、5项条件跳过；真实GPU0 CUDA/FP16反向通过。

## 3. smoke与显存

第一次3 epoch smoke在任何训练批次前因新工作树缺少AMP本地资产而人工停止，失败metadata保留在无`-r1`目录。补本地资产后，`-r1`完成3/3，CSV、best、last和外部验证完整；日志峰值14.1 GB，无OOM、NaN或Inf。

smoke的100张val在早期AP接近零，epoch2和epoch3各触发一次Ultralytics fitness-collapse恢复。训练loss和checkpoint全部有限，best内RDCI残差卷积权重绝对均值`0.000627`、最大值`0.001961`，说明支路实际参与优化。该恢复必须记录，但小子集指标不用于候选选择；正式完整数据任一组出现恢复将停止审计。

batch64固定子集1 epoch预检完成，日志峰值21.8 GB。历史证据显示完整BDD可能比固定子集再增加约1.6 GB，因此不冒险使用batch64。正式R0/R1统一为batch32/nbs64。

## 4. 冻结Stage20

| 组别 | 模型 | 唯一处理差异 |
|---|---|---|
| R0 | 官方YOLO26n-P2 | control |
| R1 | YOLO26n-P2+RDCI | 第18层P1细节/上下文残差注入 |

两组均使用BDD 70k/10k、640、batch32/nbs64、seed0、相同COCO初始化和200轮学习率计划，由运行器在epoch20保存完整optimizer/scaler/EMA后暂停。配置测试保证除`model/name`外完全相同。

R1相对R0必须满足small AP至少+5%、small AR正提升、all AP不低于-2%、medium/large AP不低于-5%，且训练健康，才允许进入Stage40。未通过即关闭，不追加变体。

## 5. 冻结证据

- smoke配置SHA256：`f765d52cb5b2dff8c02f3490e85ed3f25fc80ba735a093da2d561dc51968b0ae`
- 模型YAML SHA256：`5282d978160e210d85bbb2dffc767b23f12005f3114074efd365c4f52c7434db`
- 预训练权重SHA256：`9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`
- smoke best SHA256：`56dff3b65f78e444c55c9ff8fee104c56c2704448036d00118f4a6df98c4ec49`
- smoke last SHA256：`d92fd82d50aecceca5b0203b51b3ab891ee1eed9dd358f726fadd09b39e40fc3`
