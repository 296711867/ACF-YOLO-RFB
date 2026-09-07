# VisDrone2019-DET正式实验协议

> 工作包：`VISDRONE-PROTOCOL-FREEZE`
> 日期：2026-08-04（Asia/Shanghai）
> 状态：Stage 20 配对训练与双口径评估已完成；P2 未通过保留门，禁止续训至 epoch 100。

## 1. 实验问题

VisDrone不是BDD100K的同类别外部测试集。两者虽然都是10类，但类别语义不同：BDD的person/rider与VisDrone的pedestrian/people不等价，VisDrone还包含van、tricycle和awning-tricycle。因此本工作包不是BDD模型零样本迁移，而是在VisDrone上重新训练并比较：

1. 原生YOLO26n P3/P4/P5控制组。
2. 官方YOLO26n-P2 P2/P3/P4/P5候选组。

目标是验证官方P2对高密度无人机小目标是否稳定有效，不再测试已淘汰的SDD、DWF、SA-NWD或通道注意力。

## 2. 数据与划分

- train：6,471图、343,201实例，用于训练。
- val：548图、38,759实例，用于阶段比较和模型选择。
- test-dev：1,610图、75,102实例，公开GT，仅在最终选择完成后评估一次。
- val中small实例占68.59%，test-dev占67.69%，适合作为P2外部验证。
- 类别顺序固定为`pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor`，不得与BDD类别ID混用。

## 3. 初始化与公平边界

- 两组均从原始COCO `upstream/yolo26n.pt`开始。
- 原生组直接加载`yolo26n.pt`；P2组使用官方`yolo26-p2.yaml`并以同一权重迁移共享层。
- 禁止加载BDD100K训练后的P2 checkpoint。BDD与VisDrone恰好都是10类，形状会允许错误迁移语义不匹配的分类头，属于隐蔽数据协议错误。
- 除model/name外，两组固定相同data、imgsz、epochs、batch/nbs、seed、增强、优化器、patience和设备。
- 初始冻结640、batch32/nbs64；P2全量1-epoch预检发生TaskAlignedAssigner OOM并回退CPU，按规则两组统一重冻结为batch16/nbs64。seed0、100-epoch调度不变，在epoch20保存完整checkpoint后暂停。GPU1有外部任务，只使用物理GPU0。

冻结配置：

- `configs/train/visdrone-yolo26n-640-e100-stage20-b16.yaml`
- `configs/train/visdrone-yolo26n-p2-640-e100-stage20-b16.yaml`

## 4. 双评估口径

内部配对口径：Ultralytics val与现有COCO尺寸脚本，报告P/R/mAP50/mAP50-95及all/small/medium/large AP/AR。两组使用完全相同的转换标签，因此适合结构消融。

官方口径：当前YOLO转换跳过原始标注中的ignored regions，而VisDrone官方评估会对这些区域特殊处理。正式论文结果必须新增预测导出器并接入官方DET评估工具；未完成前只能称“项目内部VisDrone口径”，不能和论文榜单直接横向比较。

## 5. 执行顺序

1. [已完成] 数据入口、8,629图/标签、类别范围、空标签、框格式和尺寸分布复核；两组正式配置preflight通过。
2. [已完成] 从原始val/test-dev ZIP恢复官方annotations到只读评估目录，记录树哈希。
3. [已完成] 实现YOLO预测到VisDrone文本格式导出，并用合成例验证坐标、类别和置信度。
4. [已完成] 接入官方v1.0.4算法兼容Python evaluator，合成测试和548图真实格式烟测通过。
5. [已完成] batch32原生通过、P2因OOM失败；两组统一降为batch16后重新预检通过。
6. [已完成] 顺序运行两组epoch1–20，同100-epoch调度且在20保存后暂停。
7. [已完成/未通过] 完成val整体、尺寸、分类别、耗时和显存比较；P2 small AP仅+3.11%、small AR -6.92%、large AP -13.82%，不恢复到epoch100。
8. [已完成/部分终止] 两组best已在val独立复验并完成官方兼容评估；失败候选不进入test-dev，test-dev保留给最终模型一次性评估。

## 6. Stage 20保留门

P2只有同时满足以下条件才恢复到100：

- small AP50-95相对原生组提升至少5%；
- small AR50-95为正提升；
- all AP50-95相对下降不超过2%；
- medium/large AP50-95不下降超过5%；
- 无OOM、NaN、Inf、自动恢复或协议差异；
- 训练时间和显存增量完整记录，且batch没有在单一run内自动改变。

若失败，停止P2的VisDrone长训练并保留原生YOLO26n结果；若通过，用户可授权两组从完整epoch20 checkpoint直接续到100。最终结论以test-dev官方口径为主、内部尺寸口径为辅。

## 7. Preflight证据

2026-08-04两组最终在commit `175f6f4`通过受控启动器preflight，正式训练在clean commit `adf4936`启动；物理GPU0为RTX 3090，目标run目录互不冲突。

- 数据YAML SHA256：`8102e31a68f4f272f22d83068b90e8f575ab12fa4fdc9f7e4134596c511cc0e2`。
- 原生配置SHA256：`7abe9b8041007fcb13853c8c619f9a48e203d5acf7cfbdcc62d8b2f9dd4d3408`。
- P2配置SHA256：`410c1014bb1ad3c93de199cd3df610736fcb8edf8b749bb0c09f20aca422de9c`。
- P2 YAML SHA256：`0dbc2fa3dfede5edcda2566ab8c7ea150ad066ea29d79e68d6c4da2bab476279`。
- COCO初始化权重SHA256：`9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`。

Stage 20完整证据与停止决定见`results/reports/visdrone/visdrone_stage20_report.md`。
