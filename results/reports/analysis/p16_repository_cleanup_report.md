# P16仓库生命周期清理报告

> 执行日期：2026-08-08（Asia/Shanghai）

## 1. 清理结果

项目总占用由34,518,523,904字节降至34,055,340,032字节，释放463,183,872字节（441.7MiB）。剩余主要占用为数据集约19.3GiB、规范结果约4.5GiB、训练runs约702MiB和历史证据工作树约7.2GiB。

已删除：

- 245个`__pycache__`、`.ruff_cache`或`.pytest_cache`目录；
- P12的`train-thumbnail-cache.npz`和`dev-thumbnail-cache.npz`，合计约166.9MiB；
- P8、P9、P11、P15四个已合并且没有独有实验产物的工作树；
- 上述四个工作树对应的已合并本地分支。

## 2. 证据核对

P9运行时JSON保留历史绝对路径，不改写原始机器记录。移除P9工作树前，三个输入与根目录规范副本逐项核对一致：

| 输入 | 根目录规范副本 | SHA256 |
|---|---|---|
| 路由checkpoint | `results/pgsr-router-prep/router-bce-seed0.pt` | `544a3f47997bac89a44c7dc084f3a9137dbb9c09eedb10c446757372dbc0d7cf` |
| 验证集路由score | `results/perspective-router-final/val-seed0-scores.jsonl` | `9df0e4a9b53fc03b1f342eabb2ad8b1fdf3bce5298235d046bb01e2a7bafacb8` |
| 冻结manifest | `configs/eval/pgsr-final-val-10000.txt` | `bfcf0a831655f4aaba057b746a49cc3ae982041b4a89f5245a02882eff03ee34` |

P12删除的NPZ只含可重建缩略图张量。其JSON元数据、seed0 checkpoint、指标、哈希和失败停止门仍保留；此次清理不改变实验结论。

## 3. 明确保留

- P2/P4/P5/P6：仍含根目录没有的唯一训练日志、预测、smoke记录或路径迁移关系。
- P7：强制外部证据审计根，并保留正式路由预测及早期路由准备证据。
- P13：保留四份未入Git的VisDrone逐图分辨率预测。
- P14：保留三分辨率train预测、集合缓存和三seed checkpoint。
- `data/`、正式checkpoint、唯一test-dev结果、manifest和论文机器证据均未删除。

这些保留项不能仅凭“路线已失败”删除；后续必须先生成逐文件SHA256清单并迁移唯一证据。

## 4. 清理后验证

- 100项单元测试：95项通过，5项CUDA条件跳过；
- 默认论文声明审计通过；
- 强制P7外部证据论文声明审计通过；
- 仓库布局与628份Markdown链接审计通过；
- `pip check`和`git diff --check`通过。
