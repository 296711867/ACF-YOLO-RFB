# Scout检测集合分辨率价值路由协议

> 工作包：P14-SCOUT-DETECTION-VALUE-ROUTER  
> 方法简称：ScoutDet-RV  
> 冻结日期：2026-08-08（Asia/Shanghai）  
> 状态：数据推理与训练前冻结

## 1. 假设

P13证明VisDrone上960相对640的小目标收益显著且强度不均。P14检验：一次低成本320 scout检测产生的目标集合，是否比RGB缩略图和简单统计更可靠地预测“该图改用960而不是640”的边际价值。

本方法不读取RGB缩略图。它将scout的候选框位置、尺度、类别、置信度和密度编码为无序集合，用小型置换不变网络回归连续分辨率效用。三seed模型的最终固定分数为预测均值减总标准差，在25%预算下优先选择高价值且不确定性较低的图像。

## 2. 原创边界

RANet已提出分辨率自适应推理[@yang2020ranet]；DynamicDet使用多尺度特征路由和检测损失构建动态检测器[@lin2023dynamicdet]；DualRes进一步验证动态检测的部署与实际速度[@elhassani2026dualres]。EERO把早退转化为带拒绝选项和固定预算校准的问题[@valade2025eero]；Kassem-Sbeyti等研究了检测失败识别中的成本敏感不确定性阈值[@kassem2024cost]。

因此本文不声称首创动态分辨率、早退、集合编码、不确定性估计或预算路由。候选贡献限定为：针对道路小目标的“320 scout检测集合 -> 640/960边际价值”监督、序列隔离开发协议、跨seed下置信界和完整端到端成本核验的组合。只有超过同成本强基线并通过外部停止门后，才能称为有效方法。

## 3. 冻结数据与标签

- 检测器：VisDrone原生YOLO26n seed0、100 epoch冻结`best.pt`，SHA256 `8c9deeca9eff76a5461dd4567d5544c527b9d57546140a08b0dcba979176583c`。
- 路由开发：VisDrone train 6,471图；检测器已见过这些训练图，因此只作为路由开发证据。
- 划分：从文件名第一个下划线前提取视频序列ID，共208组；按`SHA256("p14:" + sequence_id)`稳定排序，前128组优化、接着32组选择epoch、后48组内部验证。禁止图像级随机切分。
- 外部筛选：VisDrone val 548图，仅在三seed内部门通过后访问一次。
- 永久封存：VisDrone test-dev不得读取、推理或汇总。
- 目标预测：同一冻结模型、FP32、batch16、`conf=0.001`、`iou=0.7`、`max_det=300`，分别生成640与960逐图预测。
- 连续效用：960相对640在small GT上的额外类别匹配数，对IoU 0.50:0.95求和，与P13一致。

## 4. Scout输入与模型

- scout：同一冻结模型、imgsz320，其余推理参数不变。
- 每图按置信度稳定排序保留前192个检测；不足部分使用mask填充。
- 每个检测token固定为7维：归一化中心`cx,cy`、宽高`w,h`、置信度、`sqrt(area)`和裁剪后的`log(aspect_ratio)`；类别使用8维可学习embedding。
- 集合编码器：共享token MLP、mask attention pooling、归一化sum pooling和显式检测数量，输出标准化效用均值与log variance。
- 单模型上限50,000参数；损失为连续效用加权pairwise ranking加0.25倍异方差Gaussian NLL。
- seed固定为0、1、2；单seed不选择正式模型。集成分数固定为`mean(mu) - sqrt(mean(exp(logvar)) + var(mu))`，不搜索不确定性系数。

## 5. 同成本基线

所有基线只读取相同320 scout预测，不读取GT或640检测结果：

1. 文件名哈希随机；
2. 检测数量；
3. 小框置信度和；
4. 低置信度/类别熵/拥挤度固定组合启发式；
5. 对冻结聚合统计拟合的Ridge；
6. 对同一聚合统计拟合的`HistGradientBoostingRegressor`：`learning_rate=0.05`、`max_iter=100`、`max_leaf_nodes=15`、`min_samples_leaf=20`、`l2_regularization=1.0`、`random_state=0`。

基线的特征、正则和树参数必须在运行前写入训练脚本，不允许根据内部验证结果搜索。

## 6. 阶段门

### A. 工程门

- 缓存图像顺序、类别范围、mask、效用和序列划分全部核验；
- 模型置换不变、空集合有限、FP32梯度和序列化通过；
- 参数不超过50,000，单批CUDA前后向通过。

### B. train内部验证门

固定25%预算，三seed必须全部完成。只有同时满足下列条件才访问val：

1. 每个seed的utility capture不低于最佳非神经基线；
2. 三seed集成下置信界的utility capture相对最佳非神经基线至少提高5%；
3. 集成选中图像的正效用比例高于最佳固定启发式；
4. 任一seed无NaN、Inf、数据泄漏或协议漂移。

失败立即关闭P14，不更换token数、模型宽度、损失权重、seed或基线参数。

### C. val一次性门

内部门通过后，在冻结548图val上一次性比较随机、最佳启发式、Ridge、HistGradientBoosting和ScoutDet-RV。ScoutDet-RV必须：

1. utility capture相对最佳非神经基线至少提高5%；
2. 混合检测small AP严格超过最佳启发式和HistGradientBoosting，small AR不低于二者；
3. 相对全640的small AP至少+5%、small AR为正、all AP不下降超过2%；
4. 学习减最佳基线的配对Bootstrap small AP区间下界大于0。

### D. 端到端运行时门

仅在val门通过后执行：320 scout后将图像互斥分组为640或960，各图只执行一个最终检测。RTX 3090、FP32、batch16、固定图像顺序、预热和三次轮换重复。总时间必须比全960至少快10%，否则不得宣称效率或精度-速度前沿。

## 7. 禁止事项

- 不访问VisDrone test-dev；
- 不在val上选择seed、预算、token数、模型宽度、损失、不确定性系数或基线参数；
- 不恢复P12 RGB缩略图CNN并换名；
- 不把GT Oracle、320 scout、现有检测器或已有动态检测思想写成原创；
- 不把内部门通过等同于最终外部验证；若P14通过，仍需未参与设计的第三数据集或真实部署证据。
