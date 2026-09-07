# ScoutDet-RV train内部筛选报告

> 工作包：P14-SCOUT-DETECTION-VALUE-ROUTER  
> 日期：2026-08-08（Asia/Shanghai）  
> 结论：三seed均超过强树基线，但集成相对增益未达到预注册5%门；在val前停止

## 1. 数据与边界

同一冻结VisDrone seed0检测器在train 6,471图上生成320/640/960预测；模型SHA256为`8c9deeca9eff76a5461dd4567d5544c527b9d57546140a08b0dcba979176583c`。三组仅改变输入尺寸，均为FP32、batch16、`conf=0.001`、`iou=0.7`、`max_det=300`。

320/640/960逐图预测SHA256分别为`7292a3694b1ae98097c9f2e564bff2195add06cc209290a796d6de9152b7069a`、`5ab94621dbbad5824c5c626a35ba14ef1fed687762e65f1093a6bf644315445e`和`0bf69e6319addc7632658e35d9b97dd218fdaabbbf073bfa3d096082f1d50909`。VisDrone val和test-dev均未读取或推理。

效用标签中正/零/负图像为5,240/647/584，均值26.4347，标签SHA256为`ed73e892ef143633d3f4587defaf0fce1a955c4c6c997a67dbf53cfe16aef7a4`。208个视频序列按冻结哈希顺序拆为128组优化、32组epoch选择和48组最终内部holdout；holdout共1,597图，25%预算精确选择400图。

## 2. 内部holdout结果

| 方法 | utility sum | utility capture | 正效用比例 |
|---|---:|---:|---:|
| 随机 | 12,445 | 0.40963 | 0.8325 |
| 检测数量 | 16,253 | 0.53497 | 0.8750 |
| small置信度和 | 23,509 | 0.77381 | 0.9675 |
| 不确定性-拥挤度启发式 | 23,521 | 0.77420 | 0.9700 |
| Ridge统计 | 24,019 | 0.79059 | 0.9650 |
| HistGradientBoosting | 24,601 | 0.80975 | 0.9675 |
| ScoutDet-RV seed0 | 25,399 | 0.83602 | 0.9775 |
| ScoutDet-RV seed1 | 25,187 | 0.82904 | 0.9725 |
| ScoutDet-RV seed2 | 25,366 | 0.83493 | 0.9725 |
| ScoutDet-RV三seed下置信界 | 25,426 | 0.83690 | 0.9775 |

三个seed均高于最强非神经基线，且集成正效用比例高于最佳固定启发式。但集成utility capture相对HistGradientBoosting只提高3.3535%，低于推理前冻结的5%门，因此整体内部门失败。

## 3. 裁决

P14在VisDrone val前停止：不生成val 320/640/960预测，不执行混合检测AP、Bootstrap或端到端运行时，也不访问test-dev。不得通过降低5%门、去掉HistGradientBoosting、选择单seed或修改不确定性系数使候选晋级。

该负结果表明检测集合确实比简单启发式更能预测分辨率价值，但4,061参数神经集合编码器相对固定强树模型的独立优势不足。它不能作为降低当前SCI拒稿风险的已验证创新点。

机器裁决位于`results/scout-router-screen/summary.json`，SHA256为`e451d741221812ed8ad5aeb2908b07b4ad3739fa4974b0de1063182e142aa80c`。逐图预测、25MB缓存和三个checkpoint按`.gitignore`保留在P14工作树，所有哈希已进入metadata和summary。
