# ACR/ACFR 论文补图（P20 §13）

> 生成日期：2026-09-17。仅使用冻结权重推理与特征 hook，未训练任何模型。
> 权重：YOLO26n（phase1 基线 200ep）、官方 P2（phase1 200ep）、ACR（phase18b r1 200ep）、ACR+ACFR（phase19a a1 200ep，即 L0 教师）。

## 案例筛选规则（冻结于渲染前）

- 匹配：类别一致、贪心 IoU≥0.5，显示阈值 conf≥0.25；
- 事件定义：ACR修复P2漏检（ACR 命中而 P2 未命中的 GT）；ACFR修复ACR漏检；ACFR新增误检（ACFR 预测无同类 GT 匹配且 ACR 同位置无误检）；共同漏检（ACR 与 ACFR 均未命中的 GT）；
- 场景为操作化代理（BDD 属性元数据未随库分发，已在图注中说明）：远距离交通标志=交通标志 GT 面积≤1024px²；夜间交通灯=含交通灯 GT 且灰度亮度<60/255；密集车辆=机动车 GT≥15；遮挡行人=≥2 个 person GT 两两 IoU≥0.15；
- 排序：场景内按事件数降序、图像名升序取前 3（不足时从全库同事件图补齐）。

## 目录

| 目录 | 内容 | 数量 |
|---|---|---|
| `five_col/` | GT \| YOLO26n \| P2 \| ACR \| ACR+ACFR 五列对比（四场景×3） | 12 |
| `events/acr_fixes_p2_*` | ACR 修复 P2 漏检（黄色高亮） | 3 |
| `events/acfr_fixes_acr_*` | ACFR 修复 ACR 漏检（黄色高亮） | 3 |
| `events/acfr_new_fp_*` | ACFR 新增误检（诚实案例，黄色高亮） | 3 |
| `events/shared_miss_*` | ACR 与 ACFR 共同漏检（诚实案例） | 3 |
| `gate_heatmaps/` | ACFR G=4 分组互补门控权重热图（sigmoid 后叠加，2 张） | 2 |

机器可读记录：`selection_manifest.json`（逐图事件计数与规则）、`render_log.json`（图清单与推理来源）。

## 推理来源说明

- P2/ACR/ACFR 列直接取冻结的正式评估逐图预测（conf 0.001 全量）过滤到 conf≥0.25，与论文表格数字同源；
- YOLO26n 列为本次 CPU 新鲜推理（conf≥0.25，640，同一 letterbox），因基线逐图预测文件未保留；
- 门控热图：hook ACFRConcat（L18）gate 末层输出，sigmoid 后上采样至 640 letterbox 画布叠加。
