# P2浅层通道注意力完整BDD Stage 20准备报告

> 工作包：`P2-EARLY-CHANNEL-ATTENTION-FULL-STAGE20-PREP`
> 日期：2026-07-30（Asia/Shanghai）
> 结论：batch64预检通过；只允许启动完整BDD 20 epoch阶段门，不自动续训。

## 1. 固定协议

- 候选：`configs/models/yolo26n-p2-identity-channel-attention.yaml`。
- 初始化：原始`upstream/yolo26n.pt`，不使用50 epoch子集best热启动。
- 数据：BDD100K完整70k train / 10k val，imgsz640、batch/nbs 64/64、seed0、AMP、GPU0。
- 与官方P2-640正式实验保持相同数据、初始化、batch、优化器、增强和seed；差异为模型及阶段预算。
- 阶段配置只写20 epoch，禁止自动进入40/60/200 epoch。

## 2. batch64显存预检

固定400/100子集运行1 epoch，完成train、val、save和best重载：

- 峰值训练日志显存20.8 GB，低于RTX 3090的24 GB。
- 360/904项预训练权重迁移正常。
- 无OOM、NaN、Inf、自动恢复或CUDA错误。
- AdamW自动解析为lr 0.000714、momentum 0.9。

预检仅用于显存和工程验收，指标不进入论文或精度比较。

## 3. 启动约束

- 正式启动器已兼容项目主工作区的`.project.git`和标准Git worktree，两种模式都要求干净树。
- 使用tmux持久化启动并固定`CUDA_VISIBLE_DEVICES=0`。
- 历史P2-640约10.51分钟/epoch，因此20 epoch预计约3.5小时。
- 完成后必须检查CSV有限性、最佳点、最近10 epoch边际收益、尺寸AP/AR和实际成本，再决定是否开展下一阶段。

## 4. 启动状态

2026-07-30 16:24，正式preflight在干净commit `12bbec1b535240a694249bc148c2c2bc77435ae6`通过，随后使用tmux `yolo26-p2-attn-stage20`在物理GPU0启动。完整70k train / 10k val扫描0 corrupt，MuSGD正常解析；epoch 1已进入训练。完整数据的实际日志显存由小子集预检的20.8 GB升至22.4 GB，物理占用约23.8/24.6 GB，目前稳定；若OOM则停止并重新冻结batch，不在同一run自动降batch。配置上限为20 epoch。

后续结果：20/20 epoch健康完成，但该配置的`epochs=20`改变了相对官方P2的学习率计划，并在正常收尾时剥离optimizer/scaler。该run不作为公平阶段对照或续训起点；完整结论见`results/reports/bdd-modules/p2_channel_attention_stage20_report.md`。
