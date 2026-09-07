# VisDrone原生/P2配对显存预检报告

> 工作包：`VISDRONE-PAIRED-PREFLIGHT`
> 日期：2026-08-04（Asia/Shanghai）
> 结论：batch32因P2 assigner OOM失败；两组统一batch16后通过，允许启动Stage 20。

## 1. 固定协议

- 数据：VisDrone train 6,471图/343,201实例，val 548图/38,759实例。
- 两组：原生YOLO26n control与官方YOLO26n-P2 candidate，不含自定义模块。
- 同源COCO `yolo26n.pt`、640、nbs64、seed0、deterministic、AMP、相同增强与优化器。
- 物理GPU0 RTX 3090；GPU1外部任务未被抢占。
- 预检为完整train/val各1 epoch，结果只作工程和预算证据，不进入论文精度表。

初始化日志：原生架构按同名类别重映射4/10个COCO分类头行并迁移612/708项；官方P2因检测层结构变为四尺度，按既有官方P2协议迁移360/902项。两组使用同一源权重，不加载BDD十类checkpoint；迁移项差异是架构变化的一部分，正式报告必须保留说明。

## 2. batch32失败证据

| 模型 | 日志峰值显存 | epoch时间 | 状态 |
|---|---:|---:|---|
| 原生YOLO26n | 约16.5 GB | 39.04秒 | 通过 |
| 官方P2 | 约22.3 GB | 107.82秒 | 失败 |

P2在训练约77%时两次触发`CUDA OutOfMemoryError in TaskAlignedAssigner, using CPU`。run随后依靠CPU回退完成，但这违反“无OOM/自动回退”的预检门，因此不能视为通过，也不能用于正式配对。

## 3. batch16重新预检

| 模型 | 日志峰值显存 | epoch时间 | CSV有限 | OOM/回退 | 结果 |
|---|---:|---:|---|---|---|
| 原生YOLO26n | 约9.03 GB | 41.98秒 | 是 | 无 | 通过 |
| 官方P2 | 约22.7 GB | 88.37秒 | 是 | 无 | 通过 |

两组均完整扫描数据、完成train/val/save/reload，无NaN、Inf、自动降batch或TaskAlignedAssigner CPU回退。P2日志峰值仍接近24 GB容量，正式Stage 20必须持续监控；任何OOM都立即停止，不在同一run内继续改变batch。

## 4. 正式配置与证据

- 原生配置：`configs/train/visdrone-yolo26n-640-e100-stage20-b16.yaml`，SHA256 `7abe9b8041007fcb13853c8c619f9a48e203d5acf7cfbdcc62d8b2f9dd4d3408`。
- P2配置：`configs/train/visdrone-yolo26n-p2-640-e100-stage20-b16.yaml`，SHA256 `410c1014bb1ad3c93de199cd3df610736fcb8edf8b749bb0c09f20aca422de9c`。
- 原生preflight best SHA256：`9058ad8ec103318135d786b3c42c6a4c540644d3b9a628ad25f929fda2e6be01`。
- P2 preflight best SHA256：`8406919d8645c0dba5c861338999dd90d76a0da7ee607802c1f7368fb0ddab24`。

commit `175f6f4`上的两份正式Stage 20配置均通过受控启动器preflight：数据YAML哈希一致，GPU0可用，目标run目录独立且尚未创建。

按1-epoch实测，Stage 20纯训练时间粗估原生约14分钟、P2约30分钟，另加验证、初始化和保存开销；两组顺序执行预计约1小时。

## 5. 决策

正式配对协议从batch32永久改为batch16/nbs64，100-epoch调度、seed和其余设置不变。下一步先启动原生control至epoch20自动暂停，再启动P2至epoch20；两者都完成后统一执行内部尺寸评估和官方兼容口径导出/评估，再决定是否直续epoch100。
