# P18A Haar小波下采样Stage20筛选报告

> 日期：2026-08-28。协议：`docs/protocols/HAAR_DOWNSAMPLE_STAGE20_PROTOCOL.md`（训练前冻结）。
> 判定：**路线关闭**——small AP 相对冻结R0为-0.89%，低于预注册关闭线（+1.0%）。

## 1. 假设与设计

BDD道路小目标的共同small漏检（52,563个）主要源于下采样信息损失。将官方YOLO26n-P2骨干最浅两处stride-2卷积（第0层P1/2、第1层P2/4）替换为固定正交Haar DWT加单一3×3融合卷积（`sdf/modules/haar_downsample.py`），其余逐层与官方相同。DWT无学习参数，无变体搜索空间。

成本审计（同工具thop）：参数+15,120（+0.57%），GMACs+0.487（+10.23%；协议预估+4~5%为分母约定错误，绝对增量与解析预测一致，偏差已在训练前记录于`smoke_record.json`）。

## 2. 执行

- 工程smoke：500图3 epoch通过，无OOM/NaN/fitness恢复。
- Stage20：BDD 70k、640、200轮计划epoch20可恢复暂停、batch32/nbs64、seed0、GPU1，共3小时43分；20/20完成，checkpoint含完整optimizer/scaler/EMA。
- 尺寸评估：`scripts/evaluate_object_sizes.py`，10k val、conf 0.001、IoU 0.7，与R0评估参数一致。

## 3. 结果

在线mAP50-95轨迹：R1因第0/1层随机初始化（R0加载预训练权重）冷启动落后-16.2%（epoch1），逐轮收窄，epoch20为0.2344，反超R0的0.2338（+0.23%），val box/cls损失均更低。

完整10k尺寸评估（R0为冻结值，`results/rdci-stage20/summary.json`）：

| 指标 | R0 | R1-HWD | 相对 |
|---|---|---|---|
| all AP | 0.235178 | 0.234633 | -0.23% |
| small AP | 0.083358 | 0.082618 | **-0.89%** |
| small AR | 0.257471 | 0.258969 | +0.58% |
| medium AP | 0.267721 | 0.270091 | +0.89% |
| large AP | 0.466453 | 0.461234 | -1.12% |

## 4. 判定与解读

预注册门：small AP ≥ +2.0%继续；[+1.0%, +2.0%)边缘；<+1.0%关闭。实测small AP -0.89%，触发关闭，all AP -0.23%与small AR +0.58%不能挽救主门。

解读：无损下采样假设在BDD-640上不成立。冷启动劣势在epoch10前还清，终点all AP达到平价，但小目标精度无净收益；收益最大的是medium（+0.89%），small AR微升说明漏检略少但小目标置信度/定位质量未改善。BDD小目标（<32²像素）在第0/1层之后的P2→P3及neck路径上的信息损失可能比最浅两层下采样更关键，或640输入下小目标信息本已不足（与PGSR分辨率结论一致：全960 small AP +14.83%远超任何结构改动）。

## 5. 处置

按协议关闭：不调融合核尺寸、不换插入层、不改变子带顺序、不叠加其他模块、不进入Stage40/200轮。代码与证据保留供复查。机器证据：`results/haar-downsample-stage20/`（summary.json、r1-size-val.json；逐图预测JSONL被Git忽略，保留在本地）。本筛选为用户授权启动的筛选级证据，导师追认前不进入论文表格。
