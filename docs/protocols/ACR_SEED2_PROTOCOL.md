# ACR seed2 复制协议

> 冻结日期：2026-09-05（Asia/Shanghai），启动前写入。
> 状态：用户于2026-09-05授权。GPU0（空闲）运行；与GPU1上的seed1并行，互不干扰，训练期不做计时基准。

## 1. 目的

与seed1协议（`ACR_SEED1_PROTOCOL.md`）同构：进一步检验ACR增益跨seed稳定性，并使论文主表可报告三seed均值±标准差（对齐VisDrone最终三seed收口的惯例）。

## 2. 冻结设计

- 配置：`configs/train/bdd100k-acr-s2-candidate-640-e200-b32.yaml`——与seed0候选逐字段一致，仅四处不同：`seed: 2`、`device: "0"`（GPU0空闲）、run名称、无`stage_stop_epoch`。
- 模型YAML、预训练权重、数据、调度、batch均不变。

## 3. 判定（预注册）

- 沿用seed1协议的分支口径：seed2相对R0-200冻结值（small AP 0.103266）为正且幅度≥+3.6% → 跨seed稳定；为正但<+3.6% → 如实报告幅度区间；为负 → 如实报告并回退叙事。
- 三seed齐后报告small AP/all AP均值±标准差（样本标准差，n=3），不挑数字、不剔除任何seed。

## 4. 禁止事项

- 不因任何seed中途曲线调整字段；结论只认200轮完整10k尺寸评估。
- 不把三seed均值写成单seed结论的替代，两者并列报告。
