# BDD100K冻结预测定性误差分析

> 日期：2026-08-05（Asia/Shanghai）
> 范围：原生YOLO26n-640与官方YOLO26n-P2-640；只读取冻结的10,000张val逐图预测、图像和标签。
> 边界：未训练、未重新推理、未访问VisDrone test-dev，案例不用于反向调参或改变模型选择。

## 1. 冻结协议

所有预测先应用置信度`0.25`，再按置信度降序执行同类别、一对一贪心匹配，IoU门为`0.50`。small GT定义为原图像素面积严格小于`32²`。P2修复指同一small GT未被原生模型匹配但被P2匹配；共同漏检指同一small GT未被任一模型匹配；P2新增误检指P2未匹配预测不存在同类别且框间IoU不低于`0.50`的原生模型未匹配预测。

候选先完整写入JSONL，再按预注册规则选图：修复和新增误检按P2置信度降序，共同漏检按GT面积升序；同分时按图像名、框坐标和类别排序。每类从排序结果依次选择前三个不同图像，不进行看图后的人工替换。两份预测SHA256与`results/bdd-paired-bootstrap/summary.json`记录完全一致。

## 2. 候选统计

| 类型 | 事件数 | 涉及图像数 | 数量最多的类别 |
|---|---:|---:|---|
| P2修复small漏检 | 6,010 | 4,007 | car 2,294；traffic sign 1,735；traffic light 1,653 |
| 两模型共同small漏检 | 52,563 | 9,037 | car 23,854；traffic sign 12,438；traffic light 9,741；person 5,037 |
| P2新增误检 | 12,948 | 6,634 | car 5,109；traffic sign 3,567；traffic light 2,774 |

在原生模型未匹配的58,573个small GT事件中，6,010个（10.26%）在相同阈值下被P2修复，其余52,563个仍为共同漏检。该比例是固定阈值下的事件描述，不是召回率或AP分解，也不能与全置信度、IoU 0.50:0.95的正式指标直接替换。

## 3. 入选案例与解释

修复案例前三名均为traffic sign，P2置信度分别为0.847、0.820和0.804。这是全候选按置信度排序自然产生的结果，不表示P2只改善交通标志；完整候选中还包含2,294个car和1,653个traffic light修复事件。并列图显示P2能够在白天遮挡、植被背景和夜间条件下补回原生模型未达到匹配门的标志框。

共同漏检前三名的GT面积仅为2.30、3.37和3.98像素²，分别对应traffic light、person和traffic sign。这些极窄、接近亚像素尺度的标注即使在局部放大后也几乎只形成线段，说明增加P2层仍无法消除由原始分辨率、远距离与标注几何共同造成的检测下限。共同漏检数量远高于修复数量，因此定性证据不支持“P2解决了small目标漏检”这一过强结论。

P2新增误检前三名置信度为0.914、0.899和0.873，类别为两个bus和一个truck；图中包括重复/竞争框以及雨夜大车辆边界混淆。全候选则主要来自car、traffic sign和traffic light。P2带来的召回改善伴随额外未匹配预测，正式结论仍应以完整AP/AR和Bootstrap为主，而不是由单类案例决定。

## 4. 可复查产物

- `scripts/analyze_bdd_errors.py`：哈希校验、匹配、分类、排序、选图和PNG/PDF生成。
- `tests/test_bdd_error_analysis.py`：不依赖BDD数据的同类一对一匹配、三类事件和排序合成测试。
- `results/bdd-error-analysis/manifest.json`：阈值、输入及入选图像/标签哈希、候选计数、入选事件和全部输出哈希。
- `results/bdd-error-analysis/*.jsonl`：三类完整候选清单。
- `results/bdd-error-analysis/p2-repaired-small.{png,pdf}`、`shared-small-miss.{png,pdf}`和`p2-added-false-positive.{png,pdf}`：GT、YOLO26n和YOLO26n-P2并列图。

图中黄色框为当前排序事件，绿色框为匹配框，红色框为未匹配预测，白色虚线为上下文GT。为使small目标可辨认，三个模型列对同一案例使用完全相同的确定性局部视野。
