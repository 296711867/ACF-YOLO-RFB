# 基于YOLO26高分辨率特征的道路小目标检测研究

> 状态：无编号论文整合稿，等待学校模板后映射章节、图表编号与参考文献样式。
> **写作基线（2026-09-07）**：本稿成稿于P15时期，**尚不含P18 ACR最终结果**。所有数字以`docs/paper/RESULTS.md`为唯一权威；ACR章节（设计、200轮主表、widen对照、三seed、推理速度）待导师追认创新口径后按RESULTS.md§1.10并入，并入前不得手抄中间数字。已归档的前身稿见`docs/paper/archive/`。
> 证据边界：正文只使用冻结实验、机器结果和已核验引用，不引入新实验结论。

## 摘要

道路场景中的行人、交通标志和远距离车辆通常只占据少量像素，连续下采样容易削弱其纹理与边界信息。针对这一问题，本文以端到端YOLO26n为基线，围绕高分辨率特征的有效性、适用边界和计算代价开展可复现实证研究。首先，在BDD100K上对原生P3/P4/P5结构与官方P2检测结构进行同数据、同初始化和完整200轮训练比较，并以尺寸AP/AR、图像级配对Bootstrap和确定性误差分析验证结论。其次，按照预注册阶段门评估细节保持下采样、动态加权融合、尺度感知损失、浅层通道注意力、跨层蒸馏、细节上下文注入和透视约束复制粘贴等候选，完整保留未通过候选的负结果。最后，在固定25%高分辨率预算下依次评估BDD透视分带路由、VisDrone分辨率Oracle上限和基于320 scout检测集合的价值路由，并将学习方法与随机、启发式、Ridge和梯度提升树基线比较。

实验表明，官方YOLO26n-P2在BDD100K上的all AP和small AP分别为0.2682和0.1015，相对原生模型提高5.01%和14.32%，对应配对Bootstrap增量区间均严格大于0。BDD学习路由在完整10,000张验证图像上的all AP、small AP和small AR达到0.2798、0.1155和0.3189，相对全640提高4.31%、13.75%和8.76%；但small AP略低于等预算粗检启发式，且集成运行时比全960慢4.61%。VisDrone上，全960相对全640的small AP提高42.72%，25% Oracle保留64.77%的绝对增量，证明跨场景仍存在可集中的分辨率价值；然而ScoutDet-RV三seed集成的utility capture为0.83690，相对强梯度提升树基线0.80975仅提高3.35%，未达到预注册5%门并在验证集前停止。本文据此将贡献限定为严格预算评估、强基线比较、可复现证据链和正负结果边界，不将官方P2、Oracle或未通过门槛的候选表述为原创有效模型。

## 关键词

道路小目标检测；YOLO26；高分辨率特征；自适应分辨率；配对Bootstrap；可复现实验

## Abstract

Small road objects, such as distant vehicles, pedestrians, and traffic signs, occupy few pixels and are vulnerable to information loss during feature downsampling. This thesis presents a reproducible empirical study of the effectiveness, limitations, and computational cost of high-resolution features in end-to-end YOLO26n. Native YOLO26n and the official P2 configuration are first compared on BDD100K under identical initialization and full 200-epoch training. The comparison is supported by size-specific AP/AR, paired image-level bootstrap, and deterministic error analysis. Architecture, loss, distillation, augmentation, and adaptive-resolution candidates are then evaluated with preregistered stage gates. Under a fixed 25% high-resolution budget, learned routers are compared with random, heuristic, linear, boosted-tree, and Oracle alternatives.

The official YOLO26n-P2 achieves 0.2682 all AP and 0.1015 small AP on BDD100K, improving the native model by 5.01% and 14.32%, respectively; both paired-bootstrap intervals are strictly positive. The BDD learned router reaches 0.2798 all AP, 0.1155 small AP, and 0.3189 small AR, but its small AP is slightly below the equal-budget heuristic and its integrated runtime is 4.61% slower than full-960 inference. On VisDrone, full-960 raises small AP by 42.72% over full-640, while a 25% Oracle retains 64.77% of the absolute gain. Nevertheless, the three-seed ScoutDet-RV ensemble reaches 0.83690 utility capture, only 3.35% above the 0.80975 boosted-tree baseline, and is stopped before validation by the preregistered 5% gate. The contribution is therefore a rigorous budget-controlled evaluation with strong baselines, traceable evidence, and explicit negative boundaries rather than a claim of a universally superior new detector or router.

## Keywords

road small-object detection; YOLO26; high-resolution features; adaptive resolution; paired bootstrap; reproducible evaluation

## 研究背景与问题

自动驾驶和道路监控需要同时识别近距离大目标与远距离小目标。小目标具有像素少、外观信息弱、遮挡比例高和类别间差异不充分等特点；当检测网络以较大步长连续下采样时，这些信息更容易在深层特征中丢失。多尺度特征金字塔通过融合高层语义与低层空间信息缓解这一问题[@lin2017fpn]，但增加高分辨率特征层也会提高训练显存、计算量和推理成本。

本文不以堆叠模块或追求单一最高点估计为目标，而是回答以下问题：

1. 官方P2高分辨率检测层能否在完整道路数据上稳定改善小目标，并付出怎样的计算代价？
2. 结构、损失、蒸馏和数据增强候选的短跑收益能否在公平预算与完整评估下保留？
3. 在固定高分辨率复检预算下，学习路由是否优于全640和合理的简单启发式？
4. 高分辨率价值能否在VisDrone上复现，学习路由在强非神经基线下是否仍具有独立必要性？

围绕这些问题，本文建立数据审计、受控训练、尺寸评估、阶段停止门、配对Bootstrap、冻结预测误差分析和运行时测量组成的证据链。研究重点是结论是否可复查以及边界是否明确，而不是把失败候选重新包装为最终网络。

## 相关工作

FPN利用自顶向下路径和横向连接融合不同尺度特征[@lin2017fpn]，EfficientDet进一步采用可学习归一化权重构建双向特征金字塔[@tan2020efficientdet]。针对低分辨率目标，SPD-Conv通过空间到深度重排减少步长卷积造成的信息损失[@sunkara2022spd]，DySample以学习采样位置实现轻量内容感知上采样[@liu2023dysample]，NWD则以归一化高斯Wasserstein距离改善微小框定位[@wang2021nwd]。这些工作分别对应本文受控验证的下采样、融合和定位损失方向，但相关思想本身不属于本文原创。

检测蒸馏研究已经覆盖尺度解耦、前景和背景加权、跨检测头监督以及高低分辨率对齐[@zhu2023scalekd][@yang2022fgd][@wang2024crosskd][@qi2021msad]。因此，本文的P2到P3实验只检验特定YOLO26双分支条件下的跨层桥接和连续small权重，不把跨尺度蒸馏概念本身作为创新主张。QueryDet使用级联稀疏查询降低高分辨率检测成本[@yang2022querydet]；RANet、DynamicDet与DualRes分别覆盖分辨率自适应、动态检测架构和部署导向的动态路由[@yang2020ranet][@lin2023dynamicdet][@elhassani2026dualres]。因此，本文不声称首创动态分辨率，而将重点放在道路小目标的边际分辨率效用、强基线和停止门。

小目标复制粘贴可增加训练图像中的小实例频次[@kisantal2019augmentation]，交通场景与道路损伤研究进一步引入类别和透视约束[@li2022traffic][@siripathitti2024precision]。本文据此验证纯框同图PASCP，但其完整阶段结果没有改善small AP/AR。近期YOLO26采用端到端双分支训练和小目标标签分配设计[@jocher2026yolo26]；SDD-YOLO和工业回收研究也强调高分辨率小目标检测及精度成本联合评估[@chen2026sddyolo][@messai2026recycling]。不同工作的数据、类别和指标并不一致，本文不进行跨论文数值排名。

## 方法与实验体系

### 基线与官方P2结构

本文使用YOLO26n作为原生基线。原生模型输出P3/8、P4/16和P5/32三个检测尺度；官方P2配置增加P2/4检测输出，使小目标在进入检测头前保留更高空间分辨率。项目直接复用冻结上游源码中的官方配置，不改变其端到端检测头，也不把P2命名为本文原创模块。

BDD100K实验使用70,000张训练图像和10,000张验证图像[@yu2020bdd100k]。原生YOLO26n和官方P2均使用640输入、batch 64、seed 0、200 epoch和同一COCO预训练初始化。所有正式指标由冻结最佳权重在完整验证集上独立推理得到。

### 候选筛选与停止门

研究过程验证了SDD细节保持下采样、DWF动态加权融合、SA-NWD尺度感知定位损失、融合前浅层通道注意力、P2到P3跨层蒸馏、RDCI细节上下文注入和PASCP透视约束复制粘贴。候选先通过形状、梯度、混合精度、序列化和小数据健康测试，再进入同预算阶段筛选。预注册核心门要求small AP相对提升至少5%、small AR为正、all AP下降不超过2%，同时避免中大目标和计算成本出现不可接受退化。未通过门槛的候选停止训练，不在同一验证集继续搜索超参数。

### 透视分带分辨率路由

道路图像中的目标尺度与垂直位置存在统计关联。透视路由从冻结P2模型第19层的32通道特征提取描述子：将特征沿纵向等分为四带，分别计算各带的通道均值和最大值，再由8,769参数的多层感知机预测图像从960复检中获得小目标收益的可能性。所有图像先执行640推理，固定选择得分最高的25%图像执行960全图复检，并用960结果替换对应640结果。

路由训练使用BDD train中互斥的5,000图清单，开发使用独立500图清单。监督标签仅由冻结640/960预测与GT离线计算，检测器参数不更新。seed 0、1、2通过开发门后，正式评估保持最先训练的seed 0，并在协议冻结后一次性访问完整10,000图验证集。正式结果产生后不更改随机种子、预算、特征层、分带数或损失。

### 指标、统计与运行时

内部尺寸评估采用COCO式101点AP，IoU阈值为0.50至0.95，maxDet为300；small、medium和large分别对应面积小于32²像素、32²至96²像素和不小于96²像素。BDD模型与路由比较均以相同图像执行1,000次配对非参数Bootstrap，每次以图像为单位有放回重采样，并重新计算全局置信度排序和AP。该区间衡量验证图像采样不确定性，不替代训练多seed，也不构成等价性检验。

集成运行时在同一RTX 3090、FP32、显式batch16和固定500图上比较全640、全960、随机25%、粗检启发式25%和学习路由25%。各策略先预热，再以轮换顺序重复3次；计时覆盖路径输入、推理、路由、960复检、CPU结果物化和结果替换，不包含模型加载与预热。

### VisDrone外部实验

VisDrone2019-DET包含6,471张训练图像、548张验证图像和1,610张test-dev图像[@du2019visdrone]。由于其10类语义与BDD100K不同，所有VisDrone模型从原始COCO权重重新训练，不迁移BDD分类头。公平筛选固定640、batch16、nbs64、seed0和相同100轮调度；阶段门后仅原生YOLO26n继续至100轮，并使用seed 0、1、2重复。test-dev只在模型和推理配置冻结后由seed 0访问一次，随后永久封存。

### 跨数据集分辨率上限与ScoutDet-RV

为避免直接迁移BDD类别语义，VisDrone分辨率实验使用该数据集已冻结的原生YOLO26n seed0模型。P13在548张val上只改变输入尺寸，比较全640、全960和25% GT Oracle；该val参与过历史模型选择，因此结果只作为跨数据集假设开发证据，不称为未见测试集验证。P14随后仅使用VisDrone train：先执行320 scout检测，将最多192个候选框编码为置换不变集合，由4,061参数ScoutDet-RV预测960相对640的连续small效用。208个视频序列按哈希分为128组优化、32组epoch选择和48组内部holdout；三seed下置信界与随机、固定启发式、Ridge和固定参数HistGradientBoosting比较。预注册要求集成utility capture相对最强非神经基线至少提高5%，失败则不访问val。

## 实验结果

### BDD100K基线与P2收益

| 模型 | 参数量 | GFLOPs | all AP | small AP | small AR | 训练时间 |
|---|---:|---:|---:|---:|---:|---:|
| 原生YOLO26n | 2,507,700 | 5.8 | 0.2554 | 0.0888 | 0.2371 | 17.08 h |
| 官方YOLO26n-P2 | 2,518,352 | 7.5 | 0.2682 | 0.1015 | 0.2932 | 35.03 h |

官方P2相对原生模型的all AP提高5.01%，small AP提高14.32%，small AR提高23.62%；参数量仅小幅增加，但GFLOPs增加约29.3%，训练时间约为原生模型的2.05倍。1,000次配对Bootstrap中，all AP增量95%区间为[+0.00998,+0.01563]，small AP为[+0.00992,+0.01621]，两者1,000次重采样均为正。这支持P2在当前BDD验证样本上的收益，但BDD完整训练只有seed 0，不能据此声称训练随机性已经消除。

BDD尺寸AP比较图使用资产`bdd-size-ap-comparison`。它同时展示原生模型、官方P2和完整训练的浅层通道注意力候选，避免用短子集筛选结果替代正式结论。

### 候选负结果

融合前浅层通道注意力在完整200轮训练中small AP仅提高0.14%，small AR下降1.70%，短子集信号没有复现。P2到P3蒸馏的D2和D3在epoch20的small AP仅提高1.43%和1.42%，D3没有体现双分支共识的增量价值。RDCI的all AP提高1.20%，但small AP仅提高1.30%、small AR下降0.56%；PASCP的all AP、small AP和small AR分别下降1.32%、3.70%和6.42%。SDD、DWF和SA-NWD也因整体退化或小目标收益不足而关闭。

这些结果说明工程可运行不等于精度假设成立，也说明短子集和早期阶段信号不能替代完整数据与预注册门。负结果保留配置、checkpoint、尺寸指标和报告，用于界定方法适用范围，而不再组合成所谓最终YOLO26-SDF模型。

### 固定预算分辨率路由

| 策略 | all AP | small AP | small AR | large AP |
|---|---:|---:|---:|---:|
| 全640 | 0.2682 | 0.1015 | 0.2932 | 0.5195 |
| 全960 | 0.2894 | 0.1292 | 0.3403 | 0.4878 |
| 随机25% | 0.2746 | 0.1069 | 0.3040 | 0.5204 |
| 粗检启发式25% | 0.2807 | 0.1162 | 0.3158 | 0.5168 |
| 学习路由25% | 0.2798 | 0.1155 | 0.3189 | 0.5138 |

学习路由相对全640的all AP、small AP和small AR分别提高4.31%、13.75%和8.76%。配对Bootstrap中，all AP差值95%区间为[+0.00942,+0.01338]，small AP为[+0.01102,+0.01693]，说明相对全640的正收益得到验证样本层面的支持。

但是，粗检启发式的small AP为0.1162，略高于学习路由的0.1155；学习路由减启发式的all AP区间[-0.00666,+0.00110]和small AP区间[-0.00282,+0.00170]均跨0。现有样本既不能证明学习头优于启发式，也不能证明二者等价。冻结门要求学习路由的small AP和small AR点估计都严格超过启发式，因此该门判定失败，不能在结果产生后放宽。

运行时结果进一步限制了方法主张。全640、全960、随机25%、启发式25%和学习25%的平均总时间分别为2.8839、3.6136、3.8250、3.7723和3.7802秒。学习路由比全960慢4.61%，三个25%策略都没有形成速度优势，原因是串行执行全640后还要对部分图像再次执行960。因而该方法只能表述为固定复检预算下的精度候选，不能表述为加速或更优精度速度前沿。

### VisDrone稳健性与泛化

VisDrone阶段筛选中，官方P2相对原生control的small AP仅提高3.11%，small AR下降6.92%，large AP下降13.82%；类别频率权重的small AP仅提高2.42%，small AR下降4.69%。两者均未通过门槛，最终保留原生YOLO26n。

冻结配置下，seed 0、1、2的all AP均值与样本标准差为0.1784±0.0018，small AP为0.0983±0.0007，small AR为0.2205±0.0016。small AP和small AR的变异系数分别为0.69%和0.74%，表明验证集核心结论对训练随机性较稳定。对应图件为`visdrone-training-curves`和`visdrone-multiseed-size-errorbars`。

模型冻结后，seed 0在test-dev上的all AP从0.1804降至0.1432，small AP从0.0985降至0.0650，降幅34.03%。official-compatible AP仅下降3.44%，原因是ignored-region处理和聚合方式不同，不能与内部尺寸AP混用。相关图件为`visdrone-val-test-size-gap`和`visdrone-official-class-gap`。test-dev结果只用于一次性外部描述，未用于继续调参。

### VisDrone分辨率价值与强基线路由

冻结VisDrone模型由640提高到960后，all AP从0.180351提高到0.217306，small AP从0.098479提高到0.140552（+42.72%），small AR从0.221136提高到0.286036。固定25% GT Oracle达到all AP 0.201256、small AP 0.125729和small AR 0.261011，保留全960相对640的64.77% small AP绝对增量。这证明VisDrone中存在可集中的分辨率价值，但Oracle读取GT，不是推理方法。

在未访问val的序列隔离train holdout上，ScoutDet-RV seed0/1/2的utility capture为0.83602/0.82904/0.83493，三seed下置信界为0.83690；最强HistGradientBoosting基线为0.80975。三个seed均超过强树基线，但集成相对强树基线仅提高3.35%，低于预注册5%门。因此P14在val前停止，不产生混合检测AP、Bootstrap或运行时结果。该实验缩小了结论边界：检测集合包含有效信息，但当前神经集合编码器相对固定强树模型的独立必要性仍未成立。

### 冻结预测误差分析

在置信度0.25、同类别一对一IoU 0.50匹配下，BDD完整验证集包含6,010个P2修复small漏检、52,563个两模型共同small漏检和12,948个P2新增误检事件。原生模型未匹配small GT中有10.26%被P2修复，但极小目标仍是共同失败的主要边界，P2也引入部分重复或类别边界误检。图件`p2-repaired-small`、`shared-small-miss`和`p2-added-false-positive`按照预先冻结的排序规则选择案例，不按视觉效果替换。

## 讨论

### 高分辨率特征的价值与代价

BDD结果清楚表明P2/4特征能够改善道路小目标，但收益伴随更高GFLOPs、训练时间和显存压力。VisDrone阶段结果没有复现同样的P2优势，说明“增加高分辨率检测层”不是跨数据集自动成立的结论。数据的视角、目标密度、类别语义和尺度分布都会影响额外P2层是否值得。

### 学习路由的必要性边界

BDD学习路由显著优于全640，却没有超过利用粗检small置信度的简单规则。VisDrone上的ScoutDet-RV进一步稳定超过启发式与Ridge，但相对固定强树基线没有达到预注册幅度。两组结果共同说明固定预算分辨率选择具有价值，也说明路由论文不能只与随机或全低分辨率比较；当强基线吸收了检测数量、置信度、尺度、类别熵和拥挤度后，神经路由器的独立必要性仍需额外证据。

### 可复现性与统计边界

项目为正式结果保存配置、日志、权重哈希、逐图预测、机器可读指标和图件manifest。配对Bootstrap减少了同一验证图像比较中的采样解释歧义，但它只覆盖验证样本不确定性。BDD缺少完整训练多seed，VisDrone三seed也不能替代BDD随机性。本文不构造伪误差条，不把区间跨0解释为方法等价。

### 局限性

本文仍存在四项主要限制。第一，BDD完整主模型只有seed 0，训练随机性证据不足。第二，VisDrone补充了分辨率上限和train内部路由筛选，但val参与过历史模型选择，ScoutDet-RV又按停止门未进入val，因此仍没有未参与设计的最终外部路由验证。第三，正式运行时只覆盖RTX 3090、FP32、batch16和500图，不能外推到TensorRT、边缘设备或其他batch。第四，VisDrone test-dev只访问一次且small AP下降明显，不能依据该结果继续选择模型。后续工作应优先使用真正未见的第三数据集或真实部署证据，而不是在BDD或VisDrone继续搜索路由结构。

## 结论

本文围绕YOLO26高分辨率特征建立了道路小目标检测的受控实验与证据体系。官方YOLO26n-P2在BDD100K上取得明确且经配对Bootstrap支持的small AP收益，但增加了显著计算成本，且在VisDrone阶段实验中没有表现出普适优势。多个结构、损失、蒸馏和增强候选在公平阶段门下未达到保留标准，说明工程可行性和短跑信号不能替代完整数据验证。

透视分带学习路由在固定25%复检预算下相对全640改善了all AP和small AP，但没有超过简单粗检启发式，也未形成端到端速度优势。VisDrone进一步证明高分辨率收益在另一场景中仍有显著上限，却再次显示神经路由相对强树基线的增量不足。本文的可靠结论是：高分辨率特征对道路小目标有效，但自适应使用方式必须同时经受跨数据、强基线和完整成本核验。研究贡献主要体现在严格预算比较、可追溯证据链以及对正负结果和适用边界的完整报告，而不是提出一个已经证明全面优越的新检测器。
