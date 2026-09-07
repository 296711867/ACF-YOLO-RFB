# VisDrone类别权重单变量准备报告

> 日期：2026-08-04（Asia/Shanghai）  
> 工作包：`VISDRONE-CLASS-WEIGHT-STAGE20`  
> 状态：Stage 20与双口径评估已完成；small AP/AR门失败，路线关闭。

## 1. 选择理由

VisDrone P2 Stage20未通过：small AP仅相对+3.11%，small AR -6.92%，large AP -13.82%，且训练耗时为原生control的2.15倍。本轮不再增加P2检测头、输入分辨率或自定义网络模块，改用原生YOLO26n上的训练目标单变量。

训练集类别频率高度不均衡：car 144,866个实例，awning-tricycle 3,246个，相差44.63倍；而内部与官方评估均按类别聚合。冻结`cls_pw=0.25`，分类BCE的类别权重为`(1/count)^0.25`并归一化到均值1。四次方根抑制把最大/最小权重比限制为2.585，避免完整逆频率的44.63倍极端放大。

| 类别 | train实例 | 冻结权重 |
|---|---:|---:|
| pedestrian | 79,335 | 0.659 |
| people | 27,059 | 0.863 |
| bicycle | 10,480 | 1.094 |
| car | 144,866 | 0.567 |
| van | 24,956 | 0.880 |
| truck | 12,875 | 1.039 |
| tricycle | 4,812 | 1.328 |
| awning-tricycle | 3,246 | 1.466 |
| bus | 5,926 | 1.261 |
| motor | 29,646 | 0.843 |

## 2. 公平边界

正式候选配置为`configs/train/visdrone-yolo26n-clspw025-640-e100-stage20-b16.yaml`；工程预检使用独立的`configs/train/visdrone-yolo26n-clspw025-640-e1-b16-preflight.yaml`和输出目录。相对已完成原生control，唯一训练差异是`cls_pw: 0.25`；model、COCO初始化、data、imgsz640、batch16/nbs64、seed0、100-epoch调度、epoch20暂停、优化器、增强和设备保持一致。参数量、GFLOPs和推理路径不变。

不做`cls_pw`数值搜索，不试0.1/0.5/1.0，避免反复使用val调参。test-dev继续冻结。P2、SDD、DWF、SA-NWD、注意力和960分辨率均不混入本实验。

## 3. Stage 20保留门

候选必须同时满足：

- all AP50-95相对原生control不下降超过2%；
- small AP50-95相对提升至少3%，small AR50-95为正提升；
- bicycle、tricycle、awning-tricycle三个目标类别中至少两个AP50-95提升；
- medium/large AP50-95均不下降超过5%；
- 无OOM、NaN、Inf、CPU回退、自动恢复或协议差异；
- 训练时间不超过原生control 1.05倍。

门槛通过后才考虑从完整epoch20 checkpoint恢复到100；失败则关闭类别权重路线。正式比较继续使用独立val、COCO尺寸和官方ignored-region兼容三套口径。

## 4. 工程与预检证据

- 28项单元/集成测试通过，3项CUDA专项在沙箱测试环境按设计跳过；类别权重公式及YOLO26 one-to-many/one-to-one双分支覆盖已验证。
- 正式配置在commit `13d22ba`通过受控preflight；正式配置SHA256为`af16acfa3fbf413162d7ca71c5ff9083b7606ffd94fe95c96533d180ce83e985`。
- 全量1 epoch预检在clean commit `d0c5cf4`完成，6,471 train和548 val全部通过；日志权重逐项匹配冻结表。
- 预检峰值显存约9.03 GB，epoch时间41.9821秒，与原生control预检41.98秒相同；CSV全部有限，无OOM、TaskAlignedAssigner CPU回退、NaN、Inf或traceback。
- 预检`best.pt` SHA256为`dcae3ea7c41af493e52c8751d8c7e5dfd2cf9304b3819899ced779ecc0698905`；结果仅作工程证据，不进入精度表。

Stage20最终结果与停止决定见`results/reports/visdrone/visdrone_class_weight_stage20_report.md`。
