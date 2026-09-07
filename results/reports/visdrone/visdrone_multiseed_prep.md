# VisDrone原生YOLO26n多seed复现准备

> 日期：2026-08-04（Asia/Shanghai）  
> 工作包：`VISDRONE-NATIVE-MULTISEED`  
> 状态：seed0/1/2训练与val双口径聚合已完成，工作包关闭。

## 协议

- seed0最终模型已完成100/100，并已执行唯一一次test-dev；test-dev永久封存。
- seed1/seed2相对seed0只改变`seed`和`name`；移除epoch20工程阶段暂停，100-epoch学习率计划、模型、数据、初始化、640、batch16/nbs64、优化器、增强和patience完全相同。
- 配置：`configs/train/visdrone-yolo26n-640-e100-b16-seed1.yaml`、`configs/train/visdrone-yolo26n-640-e100-b16-seed2.yaml`。
- 只使用物理GPU0顺序运行，不抢占GPU1外部任务；任一run出现OOM、CPU回退、NaN或自动恢复即停止后续run并审计。
- 每个seed预计约65分钟，两次总计约2.2小时。

## 评估边界

每个seed完成后只使用其`best.pt`执行val独立与尺寸评估，报告三seed均值、样本标准差和单seed明细。官方兼容val可在三seed完成后统一执行。严禁再次运行或读取test-dev预测/指标用于比较。

多seed只验证冻结模型的随机性稳健程度，不再承担模型选择功能；无论seed1/seed2结果如何，不改变seed0已冻结的test-dev结论。

最终三seed报告见`results/reports/visdrone/visdrone_multiseed_report.md`，机器可读聚合见`results/visdrone-multiseed/summary.json`。
