# PASCP工程准备报告

> 日期：2026-08-07（Asia/Shanghai）  
> 结论：代码、配置、单测、真实数据抽查和3 epoch训练smoke全部通过；尚无正式精度结论

PASCP已作为默认关闭的检测增强接入`Mosaic -> RandomPerspective`之后、`MixUp`之前。全量项目回归共57项通过、5项因当前测试进程无CUDA而条件跳过；论文冻结证据审计通过。新增测试覆盖默认关闭、参数范围、流水线注册、同seed逐像素确定性、small-only、框/类别同步、尺度保持、底边垂直带和遮挡拒绝。

真实BDD固定smoke train数据抽查50个增强后样本：24个样本触发PASCP，共新增70个框，单图最多3个；所有输出框均为有限数、宽高为正且位于640像素图像边界内。按预注册方式保存前5个成功样本，绿色为原有框、红色为新增PASCP框。逐文件哈希和统计见`results/pascp-prep/visual_audit.json`。

固定400/100子集、batch32、3 epoch训练smoke已在提交`f92684c`完成。CSV恰好3行且全部有限，train box loss从`3.4869`降至`3.3020`，日志显存峰值14.5 GB；无OOM、NaN/Inf、CPU回退或数据加载错误。第3轮首次验证因极短子集AP接近零触发一次Ultralytics fitness恢复，恢复后回到第2轮指标并正常完成；该现象不得作为精度结论，正式完整数据任一组若发生恢复必须停止审计。

best/last SHA256分别为`00f7b603076a2d5ac2704f277ab07702aea99df648efd091fcba4614f6d9b05e`和`dba61e78aa84067dff71235b3177a58eb4e9a7765ba51cb1ec20693cb54e497e`，`results.csv`为`01d2b15a192af0cd03fc3b35c927920d7972a26a0bbd195c249b3de95eb9bf6d`。前两次启动分别在AMP自检联网寻找本地已有权重、寻找被Git忽略的`bus.jpg`时于训练前终止；失败元数据保留在run目录，补齐只读本地链接后第三次从头完成，没有恢复半成品训练。

这些结果只证明工程链健康，不证明PASCP提高检测精度。下一步是对冻结R0/R1配置执行preflight并顺序运行完整BDD公平Stage20。
