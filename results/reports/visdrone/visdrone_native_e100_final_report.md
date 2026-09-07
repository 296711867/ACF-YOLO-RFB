# VisDrone原生YOLO26n e100最终报告

> 日期：2026-08-04（Asia/Shanghai）  
> 工作包：`VISDRONE-NATIVE-E100-FINAL`  
> 状态：训练、val独立/尺寸/官方兼容评估和唯一一次test-dev双口径评估全部完成。

## 1. 训练与模型证据

模型为原生YOLO26n P3/P4/P5，COCO `yolo26n.pt`初始化，VisDrone train 6,471图，640、batch16/nbs64、seed0、100-epoch计划。epoch1–20在checkpoint保存后暂停，随后从完整状态恢复epoch21–100；恢复时708/708项迁移且708个实时张量逐一匹配checkpoint EMA。

- 正常完成100/100，无early stopping、OOM、TaskAlignedAssigner CPU回退、NaN、Inf或traceback。
- 训练累计约3,882.18秒（64.70分钟），日志峰值显存约9.31 GB。
- 训练CSV最佳为epoch72：P/R/mAP50/mAP50-95=`0.4495/0.34125/0.3169/0.1757`。
- 最终`best.pt` SHA256：`8c9deeca9eff76a5461dd4567d5544c527b9d57546140a08b0dcba979176583c`。
- 最终`last.pt` SHA256：`23e5753aa3ae87064dc6c6936b20f4fea9d5ee1bf1c9c403ff4fe622b1f02f4a`。
- 最终权重已按Ultralytics正常完成流程剥离optimizer；后续推理统一使用`best.pt`。

## 2. val最终结果

独立Ultralytics val（548图/38,759实例，imgsz640、batch16、conf0.001、IoU0.7、maxDet300）：

| Precision | Recall | mAP50 | mAP50-95 |
|---:|---:|---:|---:|
| 0.447 | 0.343 | 0.316 | 0.176 |

COCO尺寸口径：

| 范围 | AP50-95 | AP50 | AR50-95 | AR50 |
|---|---:|---:|---:|---:|
| all | 0.180351 | 0.322229 | 0.313619 | 0.543229 |
| small | 0.098479 | 0.221841 | 0.221136 | 0.454556 |
| medium | 0.267417 | 0.451940 | 0.432253 | 0.675586 |
| large | 0.427754 | 0.582502 | 0.580755 | 0.769904 |

相对epoch20，all/small/medium/large AP分别提升25.24%/34.12%/24.51%/17.72%，说明从阶段点继续训练是必要的。

官方v1.0.4算法兼容口径（原始annotations、ignored regions、maxDets500，单位%）：AP50-95 `18.9947`、AP50 `35.8250`、AP75 `17.3444`、AR100 `25.0207`、AR500 `35.0259`。

## 3. test-dev唯一一次最终结果

模型、权重和推理配置在查看test-dev前已冻结。test-dev包含1,610图/75,102实例；本次结果不得用于重新训练、选参或模型选择。

内部整体口径：

| Precision | Recall | mAP50 | mAP50-95 |
|---:|---:|---:|---:|
| 0.395 | 0.303 | 0.260 | 0.141 |

COCO尺寸口径：

| 范围 | AP50-95 | AP50 | AR50-95 | AR50 |
|---|---:|---:|---:|---:|
| all | 0.143200 | 0.261512 | 0.292862 | 0.520732 |
| small | 0.064963 | 0.152281 | 0.186573 | 0.396313 |
| medium | 0.220325 | 0.394022 | 0.424756 | 0.708824 |
| large | 0.329504 | 0.484526 | 0.545190 | 0.771762 |

相对val，test-dev all/small/medium/large AP分别下降20.60%/34.03%/17.61%/22.97%，主要泛化短板是small目标。

官方兼容口径（单位%）：AP50-95 `18.3414`、AP50 `33.4387`、AP75 `18.1345`、AR100 `27.3966`、AR500 `35.2369`。相对val官方口径，AP50-95下降3.44%，但AR500基本持平（+0.60%）；内部与官方口径因ignored-region处理和聚合逻辑不同，禁止混为同一指标。

## 4. 原始证据

| 文件 | SHA256 |
|---|---|
| `results/visdrone-native-e100/yolo26n-e100-size-val.json` | `c9f77faa85ef9bd6fd587522d3de9fbf9d2189b056d2c0d46c0e20dba158d49c` |
| `results/visdrone-native-e100/yolo26n-e100-official-compatible-val.json` | `e05b3b87242568695e03c5405faef3af1ae4ecb70f93069d603298ab91e0a72e` |
| `results/visdrone-native-e100/yolo26n-e100-size-test-dev.json` | `28af4f170372c2f449390b0fc46dbb87dfe476c586ec8a208ea8db00f96626e0` |
| `results/visdrone-native-e100/yolo26n-e100-official-compatible-test-dev.json` | `6a49d098a1b75ce2a07f89409fa2e5143b69356b90fbc51aac9262a325b5842b` |

官方格式预测树SHA256：val `9e9d12194bd2c0136f9304cdf8ee0b1868a236de0c3e62d6666f1cc8bc6e91c0`，test-dev `adfc0b9b2b19426255616112ed4d88e53cdf206e0cdc754f43d1eb4c083fe34f`。

## 5. 决策与后续边界

1. seed0原生YOLO26n VisDrone模型正式冻结，不再训练或依据test-dev修改。
2. test-dev已完成协议规定的唯一一次最终访问，从现在起永久封存。
3. 后续若做seed1/seed2稳健性复现，配置除seed/name外不得改变，只报告train/val；不得再次查看test-dev。
4. 下一工作包为整理`docs/paper/RESULTS.md`、论文表格与误差分析，并决定是否承担两个额外seed约2.2小时的训练成本。
