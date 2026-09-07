# VisDrone2019-DET全量基础统计

> 生成日期：2026-08-04（Asia/Shanghai）
> 来源：已审计YOLO标签与原图尺寸；4个完全重复框已在转换阶段移除。

## 汇总

| split | images | instances | empty | instances/image | small | medium | large | small ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 6,471 | 343,201 | 0 | 53.036 | 207,602 | 116,619 | 18,980 | 60.49% |
| val | 548 | 38,759 | 0 | 70.728 | 26,586 | 11,105 | 1,068 | 68.59% |
| test-dev | 1,610 | 75,102 | 0 | 46.647 | 50,837 | 21,856 | 2,409 | 67.69% |

尺寸口径为原图像素面积：small `<32²`、medium `32²–96²`、large `>=96²`。VisDrone显著偏向小目标，且单图目标密度高于BDD100K，因此正式batch先冻结为32并要求P2显存预检。

## 分类别实例

| class | train | val | test-dev |
|---|---:|---:|---:|
| pedestrian | 79,335 | 8,844 | 21,006 |
| people | 27,059 | 5,125 | 6,376 |
| bicycle | 10,480 | 1,287 | 1,302 |
| car | 144,866 | 14,064 | 28,074 |
| van | 24,956 | 1,975 | 5,771 |
| truck | 12,875 | 750 | 2,659 |
| tricycle | 4,812 | 1,045 | 530 |
| awning-tricycle | 3,246 | 532 | 599 |
| bus | 5,926 | 251 | 2,940 |
| motor | 29,646 | 4,886 | 5,845 |

## 使用边界

- train仅用于训练，val用于阶段门、模型选择和超参数冻结。
- test-dev GT公开，但只在最终协议冻结和模型选择完成后评估一次，不参与调参。
- 当前YOLO标签过滤了原标注中`score=0`的ignored regions。Ultralytics指标可用于本项目内部配对比较，但不能直接冒充VisDrone官方榜单指标。
- 论文对外结果必须从原始ZIP恢复annotations，导出官方VisDrone检测格式并使用官方评估工具处理ignored regions。
