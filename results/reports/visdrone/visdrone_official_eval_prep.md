# VisDrone官方评估链准备报告

> 工作包：`VISDRONE-OFFICIAL-EVAL-PREP`
> 日期：2026-08-04（Asia/Shanghai）
> 结论：官方格式导出和v1.0.4算法兼容评估链已在本机闭环；允许进入配对显存预检。

## 1. 官方来源与原始标注

- 官方工具：`https://github.com/VisDrone/VisDrone2018-DET-toolkit`，codeload ZIP SHA256为`30298f4e0b56dcb5a49ab4a0d790b2c3329da5fa4391757920b74294351ff3b6`。
- 工具README版本为1.0.4，输出AP@[0.50:0.95]、AP50、AP75及AR@1/10/100/500，maxDets为500。
- val原始annotations从已审计ZIP恢复548份，排序文件名+内容树SHA256为`50d7697b3a89a0c682678d0df92dc3e6978f915dad65b437cda37b58c56b9b65`。
- test-dev恢复1,610份，树SHA256为`f66f642d9c601259879a92eb3c6043f68ed8ccef0ab177e5bc65179f219d3c8e`。
- val原始标注含1,378个class 0 ignored regions和32个class 11 others；这些信息在YOLO训练标签中不存在，因此必须保留双评估口径。

## 2. 本机实现

- `scripts/export_visdrone_predictions.py`：输出每图一个8列文本，XYXY转原图像素XYWH，模型类0–9转官方类1–10，全局按score降序并限制500框；同时记录模型、设置和预测树哈希。
- `scripts/evaluate_visdrone_official.py`：逐逻辑移植官方MATLAB v1.0.4的`dropObjectsInIgr`、`compOas`、`evalRes`、`VOCap`和`calcAccuracy`。
- 服务器没有MATLAB/Octave，因此当前实现准确名称为“官方v1.0.4算法兼容Python端口”。导出的文本可直接交给原版MATLAB工具做最终交叉复核；在该复核前不声称运行了原版MATLAB程序。

## 3. 测试与真实格式烟测

4个合成测试全部通过：

1. score排序、maxDet限制、XYWH和一基类别映射。
2. 单个完美预测AP/AR均为100%。
3. 高分FP排在TP之前时AP为50%，AR@1为0、AR@10为100%。
4. ignored region内高分检测被移除，不降低完美预测AP。

真实链路使用BDD官方P2十类checkpoint只做格式烟测，不作VisDrone精度结论：548图均成功导出，共247,384个检测，预测树SHA256为`e94e0f4e163428d23eb0f3d2b26facbbffce18a3a0bb6c8c3a89850b71767343`；兼容评估器完整运行并生成`results/visdrone-eval-smoke/bdd-p2-format-only-official-port.json`，文件SHA256为`d35b01b491e593cb2bf5b78206b9007cbe6169419d10151a3bfce1514f7e74c1`。

## 4. 决策

导出、ignored-region处理和指标聚合链已经可执行，训练前评估阻塞解除。下一步运行原生YOLO26n和官方P2各1 epoch、640、batch32全量VisDrone显存预检；两组都稳定后才允许启动正式Stage 20。
