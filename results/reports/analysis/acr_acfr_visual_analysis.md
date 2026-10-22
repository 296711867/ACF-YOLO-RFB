# ACR/ACFR 定性可视化分析（P20 §13，冻结权重推理）

> 日期：2026-09-17。素材：`results/paper-visuals/acr-acfr/`（26 图 + manifest/render_log）；筛选与渲染规则见该目录 `README.md`。
> 本文只读冻结预测与权重，不含任何训练或验证集选型行为。

## 1. 全库事件统计（10,000 张 BDD val，conf≥0.25、IoU≥0.5 口径）

| 事件 | 出现该事件的图像数 |
|---|---:|
| ACR 修复 P2 漏检（至少1个GT） | 6,994 |
| ACFR 修复 ACR 漏检（至少1个GT） | 5,909 |
| ACFR 新增误检（至少1个FP） | 5,154 |
| ACFR 与 ACR 共同漏检（至少1个GT） | 9,543 |

场景代理覆盖：远距离交通标志 7,425 图、夜间交通灯 2,491 图、密集车辆 2,572 图、遮挡行人（代理）532 图。

解读（与定量证据一致的方向）：

- 修复类事件广泛存在（约六到七成图像至少一处），与 ACR/ACFR 的 small AP 显著增益（+7.28%/+5.22%，200轮配对Bootstrap CI 严格>0）方向一致；
- 新增误检同样常见——ACFR 的空间门控在放大互补响应时不可避免引入少量假阳性，这与 all AP 只能写"平价"（CI 跨0）的定量结论吻合；
- 共同漏检覆盖面广（95%图像至少一处），绝大多数为极小/严重遮挡目标，是第六代改进的剩余空间，如实保留。

## 2. 图组说明

1. **五列对比**（`five_col/`，4 场景×3）：GT 列为标注；四模型列同阈值渲染。可见规律：YOLO26n 在小目标列（远处标志/行人）明显稀疏；P2 增加小目标检出；ACR 在 P2 基础上小目标更完整；ACR+ACFR 在密集与遮挡场景的检出密度进一步改善，同时偶见低置信新增框。
2. **ACR 修复 P2 漏检**（`events/acr_fixes_p2_*`）：黄色高亮 P2 未命中、ACR 命中的 GT；典型为远距离交通灯/标志与被遮挡车辆边缘。
3. **ACFR 修复 ACR 漏检**（`events/acfr_fixes_acr_*`）：黄色高亮 ACR 未命中、ACFR 命中的 GT；集中于密集车流中被吞并的小车与弱对比度目标。
4. **ACFR 新增误检**（`events/acfr_new_fp_*`，诚实案例）：黄色高亮无 GT 匹配的 ACFR 预测，多为护栏/反光镜等与交通标志混淆的低置信框。
5. **共同漏检**（`events/shared_miss_*`，诚实案例）：黄色高亮两模型均未命中的 GT，几乎全部是面积极小或高度遮挡的目标。
6. **门控热图**（`gate_heatmaps/`，2 图×4 组）：ACFRConcat 门控对 P3 语义路的增益权重（sigmoid）。不同组呈现可区分的空间模式：道路/车辆密集区与天空/背景区权重不同，与"分组互补空间门控"设计动机一致；门控在天空区域接近中性（≈基色），不产生虚假激活。

## 3. 论文使用建议

- 图注须写明筛选规则（README 已给模板），强调"按冻结事件规则选择、非人工挑选"；
- 成功与失败案例并列呈现（修复 vs 新增误检/共同漏检），与 §12.2"不得只挑成功案例"一致；
- 门控热图作为机制佐证而非独立证明（与 P19A 组件消融的定量结论 nr 显著/ng 不显著配合使用）。

## 4. 复现

```bash
PYTHONPATH=upstream:. python scripts/select_acr_acfr_visual_cases.py \
  --output results/paper-visuals/acr-acfr/selection_manifest.json
PYTHONPATH=upstream:. python scripts/render_acr_acfr_visuals.py \
  --manifest results/paper-visuals/acr-acfr/selection_manifest.json \
  --out-dir results/paper-visuals/acr-acfr
```

CPU 推理即可（YOLO26n 列约 18 图新鲜推理；其余列取冻结预测）。
