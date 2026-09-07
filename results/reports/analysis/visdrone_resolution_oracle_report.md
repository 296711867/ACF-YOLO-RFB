# VisDrone分辨率路由外部上限报告

> 工作包：P13-VISDRONE-RESOLUTION-ORACLE  
> 日期：2026-08-08（Asia/Shanghai）  
> 结论：预注册上限门全部通过，允许提出新的可学习方法协议

## 1. 边界

同一冻结VisDrone原生YOLO26n seed0 `best.pt`在548张val上分别执行640和960推理。checkpoint SHA256为`8c9deeca9eff76a5461dd4567d5544c527b9d57546140a08b0dcba979176583c`；两组均为FP32、batch16、`conf=0.001`、`iou=0.7`和`max_det=300`。没有训练、阈值搜索或test-dev访问。

VisDrone val参与过历史模型选择，因此本结果是跨数据集假设开发证据，不是未见测试集上的最终外部验证。25% GT Oracle读取标签，只表示可提升上限。

## 2. 结果

| 策略 | all AP | small AP | small AR | 相对640 small AP | 相对640 all AP |
|---|---:|---:|---:|---:|---:|
| 全640 | 0.180351 | 0.098479 | 0.221136 | - | - |
| 全960 | 0.217306 | 0.140552 | 0.286036 | +42.72% | +20.49% |
| 25% GT Oracle | 0.201256 | 0.125729 | 0.261011 | +27.67% | +11.59% |

25% Oracle精确选择137/548张图，保留全960相对640的64.77% small AP绝对增量。逐图效用为正/零/负的图像数为486/32/30，说明分辨率收益广泛存在，但收益强度高度不均匀，固定预算选择仍有价值。

## 3. 裁决

机器门控的七项条件全部为真：全960 small AP至少+5%、small AR为正、all AP不退化；25% Oracle的small AP/AR均提高、保留至少一半全960 small AP增量且all AP不退化。

P13通过只证明“VisDrone上存在可集中利用的分辨率价值”。下一工作包可以设计可学习路由，但必须与BDD P12失败的缩略图CNN不同，并单独冻结训练/验证划分、简单基线、跨数据迁移和停止门。不得使用VisDrone test-dev选型。

## 4. 机器证据

- `results/visdrone-resolution-oracle/full-640.json`，SHA256 `3725163351ddd45aa833347076bcf84b386d36da23864058c6a67df1363e69ea`。
- `results/visdrone-resolution-oracle/full-960.json`，SHA256 `21d15dd2b19dded22066845343722d7cac554a9baef4ad4ad8d051b12036a7fc`。
- `results/visdrone-resolution-oracle/oracle.json`，SHA256 `415f63e74157cb93595825a4386df75161dbf9dc79da4136931e8aff1f873bab`。
- `results/visdrone-resolution-oracle/gate.json`，SHA256 `0fdddb0942bec05f803e349cb56f1947e76263eb533f94900af009a172c7bd01`。
- 640/960预测SHA256分别为`b85327f6eab97b8b60f3196ce8fa584f6086d0bac6a9d58c3231e9f92293bea3`和`96db9b8c35767c99e84c66147106ab1bd1fff82793c8dca6bcdde3a95ccce681`，按`.gitignore`保留在P13工作树。
