# Phase 1 正式基线准备验收报告

> 工作包：`P1-BASELINE-PREP`  
> 执行日期：2026-07-25（Asia/Shanghai）  
> 结论：准备项通过；尚未启动 200 epoch 训练。

## 1. 尺寸专项评估

新增 `scripts/evaluate_object_sizes.py`，实现不依赖 pycocotools 的 COCO 风格检测评估：

- IoU：0.50:0.05:0.95。
- AP：101 点插值。
- maxDet：300。
- small：面积 `< 32²`；medium：`32²–96²`；large：`>= 96²`。
- 按类别宏平均，同时保存分类别 AP50-95、AP50、AR50-95 和 AR50。
- 推理时保存逐图原始框、类别和置信度 JSONL。

验证证据：

1. 单个完美 small 框合成测试得到 AP/AR=1.0。
2. smoke 权重在 100 张、1,931 个 GT 上完成真实推理。
3. 自定义整体 AP50-95 为 0.002604，Ultralytics val 为 0.00257，差异约 0.000034，口径一致。
4. smoke small AP50-95 为 0，符合仅训练 3 epoch 的低质量权重，禁止作为论文结论。

产物：

- `results/phase0_smoke_size_metrics.json`。
- `results/phase0_smoke_size_metrics.predictions.jsonl`。

## 2. BDD100K 全量统计

新增 `scripts/analyze_bdd100k.py`，直接扫描已审计 Lance 快照，并严格复现 YOLO 转换的退化框和重复框过滤规则。

| split | images | instances | small | medium | large | small ratio |
|---|---:|---:|---:|---:|---:|---:|
| train | 70,000 | 1,288,009 | 710,395 | 414,421 | 163,193 | 55.15% |
| val | 10,000 | 185,526 | 102,729 | 59,669 | 23,128 | 55.37% |
| total | 80,000 | 1,473,535 | 813,124 | 474,090 | 186,321 | 55.18% |

- 空标签图像：0。
- 分辨率：80,000/80,000 均为 1280×720。
- 跳过退化框：447；移除重复框：1，与转换审计完全一致。
- 已输出类别实例/图像数、天气、场景、时段、遮挡和截断统计。
- 结果目录：`results/dataset_stats/bdd100k/`。

## 3. 正式配置

冻结配置：

- `configs/train/bdd100k-640.yaml`：YOLO26n、640、200 epoch、batch 64、GPU0、seed 0。启动时 GPU1 正被外部长期任务占用，因此不抢占；GPU0 显存足以承载该 batch。
- `configs/train/bdd100k-960.yaml`：仅在 640 基线验收后运行，其余核心设置一致。
- 输出根目录：`runs/detect/phase1/`。
- 正式名称：`bdd100k-yolo26n-{imgsz}-e200-seed0`。
- `exist_ok: false`，禁止覆盖。
- `optimizer: auto` 保留官方策略；200 epoch 总迭代数超过 10,000，预期解析为 MuSGD，最终仍以 run 日志为准。

## 4. 受控启动与版本证据

新增 `scripts/run_experiment.py`。正式训练不得直接调用 `yolo train`，统一使用：

```bash
conda activate yolo26-sdf
python scripts/run_experiment.py --config configs/train/bdd100k-640.yaml --preflight
python scripts/run_experiment.py --config configs/train/bdd100k-640.yaml
```

启动器执行以下停止门：

1. 要求独立项目 Git 目录 `.project.git` 存在且工作树干净。
2. 要求 config、model、data、project、name、seed、device 等字段完整。
3. run 目录已存在时拒绝启动。
4. 训练前记录 Git commit、配置/数据/权重 SHA256、Python、平台、Ultralytics、PyTorch、CUDA 和 GPU。
5. 保存冻结配置副本；成功后记录 best/last 权重 SHA256，失败时记录异常。

服务器项目根目录的 `.git` 是只读空挂载，并非有效 Git 仓库。项目因此使用 `.project.git` 保存真实 Git 对象，查询命令为：

```bash
git --git-dir=.project.git --work-tree=. status
git --git-dir=.project.git --work-tree=. rev-parse HEAD
```

本地提交身份固定为项目自动化身份 `YOLO26-SDF Project <yolo26-sdf@localhost>`，不冒充个人作者。

## 5. 验收结论

- [x] 尺寸评估脚本完成合成测试和真实 smoke 自测。
- [x] BDD100K 全量统计落盘并与转换审计一致。
- [x] 正式 640/960 配置、命名和输出路径冻结。
- [x] 正式启动器具备 clean-tree、不可覆盖和元数据记录门禁。
- [x] 项目独立 Git 历史将在本工作包提交时建立。
- [ ] YOLO26n 640、200 epoch 正式基线尚未运行。
