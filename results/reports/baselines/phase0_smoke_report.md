# Phase 0 冒烟测试验收报告

> 工作包：`P0-SMOKE`  
> 执行日期：2026-07-25（Asia/Shanghai）  
> 结论：通过  
> 用途限制：本报告仅证明工程链路可运行，所有精度数字禁止进入论文主表。

## 1. 标签人工审计

- BDD100K：seed 0 固定抽样 200 张，生成 10 张接触表和 200 张原尺寸标注图。
- VisDrone：seed 0 固定抽样 50 张，生成 3 张接触表和 50 张原尺寸标注图。
- 人工检查结论：框整体贴合目标，未发现系统性坐标偏移、类别 ID 越界或转换方向错误。
- BDD100K 审计目录：`results/figures/label_audit/bdd100k/`。
- VisDrone 审计目录：`results/figures/label_audit/visdrone/`。

## 2. 固定 BDD100K 子集

选择方式为 seed 0 的确定性类别覆盖抽样，只用于工程冒烟。

| 划分 | 图像 | 实例 | 空标签 | small | medium | large | small 比例 |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 400 | 7,251 | 0 | 3,945 | 2,365 | 941 | 54.4063% |
| val | 100 | 1,931 | 0 | 1,061 | 656 | 214 | 54.9456% |

尺寸统计使用 COCO 面积阈值：small `< 32²`、medium `< 96²`、large `>= 96²`，按原图像素计算。

| 文件 | SHA256 |
|---|---|
| `data/BDD100K/splits/smoke_train.txt` | `c4b434fe51fa481eb314f1714c1aebf6376033f308b63ec1fae0ee7e0f2c6ea7` |
| `data/BDD100K/splits/smoke_val.txt` | `a43eaabae1f795edf5fb9472ca947734f150abd49a31e0e960e83e1acf447dec` |
| `data/BDD100K/splits/smoke_predict.txt` | `1a50db253115b1be0b6535ffcc454a2af00c0f15a601302ff0dab556da3292b1` |
| `configs/data/bdd100k-smoke.yaml` | `30f3032f881bea875a72b1bffd55927a3acf54f26cb86225795a9b04f104aef4` |
| `configs/train/smoke-640.yaml` | `fed44fe0d797a099e3ff9f49c9f9860c249d71ccd89121ca1cc64780f6ff0333` |

完整类别与尺寸统计保存在 `data/BDD100K/splits/smoke_report.json`。

## 3. 训练

命令：

```bash
MPLCONFIGDIR=/tmp/yolo26-sdf-matplotlib conda run -n yolo26-sdf \
  yolo cfg=configs/train/smoke-640.yaml
```

关键事实：

- Ultralytics 8.4.104，PyTorch 2.6.0+cu126，CUDA:0，RTX 3090 24 GB。
- 原生 YOLO26n，2,507,700 参数，输入 640，batch 32，3 epoch，seed 0。
- 预训练分类头按类别名复用 6/10 行，最终权重类别映射完整且顺序为 BDD100K 0–9。
- `optimizer=auto` 最终解析为 AdamW，lr 0.000714，momentum 0.9。
- 峰值显存约 6.1 GB。
- 训练正常退出；`results.csv` 共 3 行、39 个指标/loss 数值，全部为有限数，无 NaN/Inf。

| epoch | train box | train cls | train l1 | val box | val cls | val l1 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.40592 | 4.92896 | 0.00519 | 2.27814 | 5.03439 | 0.00574 |
| 2 | 2.41453 | 4.14748 | 0.00500 | 2.25869 | 4.93083 | 0.00593 |
| 3 | 2.39292 | 3.69271 | 0.00508 | 2.25173 | 4.67436 | 0.00592 |

训练产物实际目录：

```text
runs/detect/runs/phase0/bdd100k-yolo26n-smoke-seed0/
```

相对 `project: runs/phase0` 被 Ultralytics 追加到默认 `runs/detect` 后，形成了嵌套路径。结果完整，但正式基线配置必须修正并冻结输出规则。

| 权重 | SHA256 |
|---|---|
| `weights/best.pt` | `821ae0333b64ad289b76af68cfbbd62c77de2d6f54e8d2dedb3b86333baeb3af` |
| `weights/last.pt` | `1897562260ebcb6de4c2edc40d68706a80a28180629a1ded1e12d9dd9bbaed69` |

## 4. 独立验证与预测

独立 val 在固定 100 张验证集上正常完成：

| images | instances | precision | recall | mAP50 | mAP50-95 |
|---:|---:|---:|---:|---:|---:|
| 100 | 1,931 | 0.0831 | 0.0676 | 0.00339 | 0.00257 |

验证目录：`runs/phase0/bdd100k-yolo26n-smoke-seed0-val/`。

固定 10 张图片的标准预测在 `conf=0.01` 下正常完成，但没有检测框。这与仅训练 3 epoch 的低置信度权重一致，不视为精度验收。另以 `conf=0.00001`、`max_det=100` 执行诊断预测，10/10 图片均成功保存框和 TXT，证明推理、后处理、绘图及标签保存链路完整；该阈值输出严重过检，不具备精度意义。

- 标准预测：`runs/phase0/bdd100k-yolo26n-smoke-seed0-predict/`。
- 诊断预测：`runs/phase0/bdd100k-yolo26n-smoke-seed0-predict-diagnostic/`。

## 5. 验收结论与遗留项

- [x] train、val、predict 均正常退出。
- [x] loss 和 CSV 数值全部有限，无 NaN/Inf。
- [x] 模型类别映射严格为 0–9。
- [x] args、CSV 日志、best/last 权重、曲线和预测图均已保存。
- [x] optimizer 最终解析结果已记录。
- [x] Phase 0 工程闭环通过。

不影响本次验收、但必须在正式基线前处理：

1. 建立可引用的项目 Git commit。
2. 修正并冻结正式 run 输出目录，禁止覆盖。
3. 实现小/中/大目标专项 AP/Recall 评估。
4. 生成 BDD100K 全量类别、尺寸和场景统计。
5. AMP 自检曾尝试访问 GitHub 并超时，随后使用本地检查正常通过；正式离线训练应避免不必要的联网探测。
