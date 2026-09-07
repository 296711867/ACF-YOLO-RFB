# P2浅层近恒等通道注意力工程与smoke报告

> 工作包：`P2-EARLY-CHANNEL-ATTENTION-PREP`
> 日期：2026-07-30（Asia/Shanghai）
> 结论：工程与3 epoch smoke通过，允许进入固定子集配对50 epoch；不允许直接长训。

## 1. 最小实现

- 基座为官方YOLO26n-P2-640，保留原生损失及`nearest + concat + C3`。
- 只对进入P2 concat的浅层backbone特征做全局通道门控，不含SDD、DWF、SA-NWD或960。
- 门控为`2 * sigmoid(logit)`，1x1权重和偏置全零初始化，因此初始增益严格为1。
- 注意力与concat占用官方concat的同一层编号，避免后续head编号偏移和预训练迁移损失。

BDD100K 10类模型参数从官方P2的2,518,352增加到2,522,512，仅增加4,160（约0.165%）；日志GFLOPs仍为7.5。

## 2. 工程验收

- shape、初始逐元素等价、输入/参数梯度、state_dict序列化通过。
- GPU0 CUDA FP16 autocast反向通过，所有梯度有限。
- YAML模型只包含一个`IdentityChannelAttentionConcat`，无SDD/DWF，stride为`[4, 8, 16, 32]`。
- 原始`yolo26n.pt`迁移360/904项，与官方P2迁移360项一致。
- 模型640 forward和训练后best checkpoint重载通过。

## 3. 3 epoch smoke

- 数据：BDD100K固定400 train / 100 val，val共1,931实例。
- 协议：imgsz640、batch/nbs 32/64、seed0、AMP、物理GPU0。
- 结果：3/3 epoch完成，约0.003小时，峰值显存约13.6 GB。
- 健康性：CSV三行全部有限；无NaN、Inf、OOM、early stopping或自动恢复。
- 最终mAP50约`3.48e-5`、mAP50-95约`1.35e-5`；仅用于工程验收，不作精度结论。
- best中注意力weight/bias绝对最大值分别约0.00423/0.00412，确认门控参与训练。

run位于worktree本地忽略目录：
`runs/detect/runs/detect/phase2-smoke/bdd100k-yolo26n-p2-identity-channel-attention-640-e3-b32-seed0/`。

两次预检退出均发生在epoch 1之前：第一次为worktree相对数据路径解析，第二次为AMP检查缺少示例图；以只读绝对数据YAML和主工作区assets链接修正后，完整smoke一次通过。它们不是模型、CUDA或训练失败。

## 4. 证据哈希

| 文件 | SHA256 |
|---|---|
| smoke `best.pt` | `681a4061d2a5da565d140c5b423883cabfb3836a7d210e0f43ac84d975cfab29` |
| 模型YAML | `3e312bc0c7a9979cb0c8b614021e172acdd92d77e1f27e53ea0db52168158605` |
| smoke配置 | `9ece000e9a0999164fd46720f04f471d986abf2c0edf86d87d49b2c706efd136` |
| 模块源码 | `2e581dfb7e9be79b0736d921d8271a2ee844940682d91f14c52ef551aaf2d68d` |
| 原始`yolo26n.pt` | `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef` |

## 5. 下一步

运行同协议候选50 epoch，随后对best执行100张val独立验证和all/small/medium/large AP/AR。只有small AP至少+5%、small AR为正、all AP不低于-2%、medium/large AP不低于-5%时才保留；否则终止注意力路线。
