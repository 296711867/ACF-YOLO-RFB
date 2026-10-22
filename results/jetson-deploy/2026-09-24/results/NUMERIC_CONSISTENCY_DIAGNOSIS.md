# 步骤 D 合成输入 FAIL 的诊断记录（2026-09-24）

## 现象

冻结子集未到位前，按文档 fallback 用固定种子合成输入（`--num 20`）跑了 §16.1-4 数值一致性，
l0 与 k1 均 `status=FAIL`，失败项只有 `max_abs_diff`（≈637，判据 ≤5e-2）；`all_finite` 与
`min_detection_agreement` 均通过（20 张输入两侧全部 0 检测，一致性 1.0 为空真）。

## 诊断（用 TUM freiburg1_room 室内序列做了域外真实图复核）

对单图逐列分析 FP16 vs FP32 输出 `[300,6]` 的 `|Δ|`：

| 列 | 含义 | max \|Δ\| | 结论 |
|---|---|---|---|
| 0–3 | x1,y1,x2,y2 | 2.6e2 ~ 6.3e2 | 槽位集合错位 |
| 4 | score | **8e-4** | 两引擎完全一致 |
| 5 | class | 9 | 仅错位槽位 |

- 该图 top 分数 FP16/FP32 = 0.0469/0.0461，全部 300 槽位 score < 0.05（conf≥0.25 判 0 检测）；
- **score 列逐槽一致而坐标列整体错位** ⇒ 差异来自 top-300 槽位选择不稳定：数千个近似相等的
  极小分数在 FP16/FP32 下排序不同，选中集合不同，与数值精度无关；
- conf 降到 0.02 让 top 候选浮现后：top-1 检测 **IoU 0.996 / 0.998、类别一致、分数差 ~5e-3**，
  即存在检测内容时两引擎结果一致。

## 结论

FP16 引擎数值健全；合成输入 FAIL 是**输入域伪影**（无高置信检测 → top-k 集合不稳定，
逐元素 max|Δ| 判据在此情形不适用）。**正式 PASS/FAIL 以 bench_100 街景图（步骤 D 官方跑，
子集到位后）为准**，本文件仅作排查记录。判据适用性建议随正式结果一并提交导师：对 NMS-free
top-k 输出，`max_abs_diff` 判据建议限定在 conf≥阈值 的行上计算，或以检测一致性为主判据。

## 附带处置

- 首个 l0_fp16 引擎（2026-09-23 构建，即本文件所述 JSON 引用的 engine_sha256）加载时报
  "engine plan across different models of devices" 警告，已于 09-24 在本机重建消除
  （旧文件保留为 `engines/l0_ep200_fp16_b1.engine.pre-rebuild.bak`）。
  `results/numeric_consistency_{l0,k1}.json` 中 l0 的 engine sha256 对应重建前引擎，
  正式复跑时自然更新。

## 本日结果文件索引（只增不改）

- `results/numeric_consistency_{l0,k1}_synthetic.json`：合成输入 FAIL（本文诊断；原 `*_synthetic`
  后缀为 09-24 正式复跑前改名，内容未改动，engine sha256 对应重建前 l0 引擎）
- 诊断原始数据：`/tmp/diag_numeric_consistency_{l0,k1}.json`（TUM 图，域外，非正式）
- `results/numeric_consistency_{l0,k1}.json`：bench_100 正式复跑（09-24，判据 G1–G4 版脚本）
