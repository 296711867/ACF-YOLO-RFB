# ScoutDet-RV方法与工程准备报告

> 工作包：P14-SCOUT-DETECTION-VALUE-ROUTER  
> 日期：2026-08-08（Asia/Shanghai）  
> 状态：协议、核心模型与缓存接口冻结；尚未运行数据推理或训练

## 1. 方法

ScoutDet-RV先用冻结检测器执行320分辨率scout，将置信度最高的192个检测编码为无序集合。每个token包含归一化几何、置信度、面积、宽高比和类别embedding；共享MLP、mask attention、sum pooling与显式数量共同预测960相对640的标准化边际效用均值和log variance。

三seed最终分数固定为预测均值减总标准差，不搜索不确定性系数。模型不读取RGB缩略图，因而不是P12检测前CNN的换名版本。

## 2. 工程证据

- 默认VisDrone 10类模型参数量：4,061，低于50,000上限。
- 置换不变性、空检测集合、mask、非法类别、稳定置信度排序、几何归一化、集成不确定性惩罚和FP32梯度测试通过。
- RTX 3090 GPU0上，batch16、192 tokens完成合成FP32前后向，loss为`0.8209163`，输入梯度全部有限。
- cache按VisDrone视频序列ID分组，禁止相邻帧跨fit/holdout泄漏；208组固定前160组拟合、后48组内部验证。

协议SHA256为`511a6712c4181af3d8008fe606a903561240bd2d62eabedb7b5ef4f2eeccf301`；模型源码SHA256为`b606b21c4ecf0487e77e6ebbd7e3514c90da53bd1de81e0888814023fd917392`；缓存脚本SHA256为`02329ac9466f8f67997cad23b8befebe1733bd5aac1f171fde7ab682773943a5`。

训练前另冻结流式预测、效用标签、单seed训练与三seed停止门。流式预测器在960运行中增加严格前缀恢复保护，最终源码SHA256为`56315ecf76ad45631dc1a59770457760eaacb7c0b7db857248c5f2b7a6fb46cc`；该工程修复不改变推理参数或已完成预测。效用标签与训练脚本仍为`ab853e088f4b0367000c818e2cb6f978324bec709c5d45cf86dac8c50356c701`和`e6d08bed418f505947ecdf03d87c3db17aa8113bd420b0f62ce3a4822bcfdfa9`；最终带证据哈希的停止门脚本为`0da88e419000173a08b1e5c7ff8f5d295aa0027cdd8f9627d100b1a453df2081`。强树基线固定使用scikit-learn 1.9.0。

## 3. 决策

工程门通过，下一步允许且只允许生成VisDrone train的320/640/960冻结预测和效用标签，再实现预注册的同成本基线与三seed训练器。内部门通过前不得访问val；test-dev继续永久封存。
