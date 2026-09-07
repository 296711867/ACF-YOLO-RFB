# 数据集下载与转换审计

审计日期：2026-07-25（Asia/Shanghai）

## BDD100K

官方 Berkeley 下载门户在服务器上超时；旧 ETH Zurich 镜像域名无法解析；旧官方 S3 对象返回 404。因此采用以下可校验组合：

- 图像镜像：`Xing1210/bdd100k`（Hugging Face 国内镜像）。
- 文件：`bdd100k_images_100k.zip`，5,669,071,832 字节。
- SHA256：`afd149eadb49657ef5ad7bb6e9d77cc8b7927a576ee2b475b720028274e02784`。
- ZIP 完整性：`unzip -t` 无错误。
- 图像：70,000 train、10,000 val、20,000 test。
- 检测标注镜像：`lance-format/BDD100K-enriched`，快照 commit `d82c5188d392714ba8091d68014f7b9838ceadf2`。
- 真实标注：70,000 train、10,000 val；镜像额外提供的 500 个 `synthetic_val_*` 已排除。
- 图像与标注跨镜像核验：固定抽样 10 张覆盖 train/val，图像字节 SHA256 10/10 完全一致。
- YOLO 标签树 SHA256：`3ebbb4d0623384387535b6d4c218bae813bb49fb39c6800ef0dbb26abd3cedc3`。
- Ultralytics 全量扫描：80,000 images、80,000 labels、0 corrupt。
- 转换时跳过 447 个零面积/退化框；没有越界框；移除 1 个完全重复框。

类别顺序与实例数：

| ID | 类别 | 实例数 |
|---:|---|---:|
| 0 | person | 104,667 |
| 1 | rider | 5,170 |
| 2 | car | 816,422 |
| 3 | truck | 34,248 |
| 4 | bus | 13,281 |
| 5 | train | 151 |
| 6 | motor | 3,454 |
| 7 | bike | 8,232 |
| 8 | traffic light | 213,109 |
| 9 | traffic sign | 274,801 |

准备脚本：`scripts/prepare_bdd100k.py`。数据配置：`configs/data/bdd100k.yaml`。

### Phase 0 固定冒烟子集

- 生成脚本：`scripts/create_smoke_subset.py`，seed 0，确定性类别覆盖抽样。
- train：400 images、7,251 instances、0 empty labels、small 比例 54.4063%。
- val：100 images、1,931 instances、0 empty labels、small 比例 54.9456%。
- 训练清单 SHA256：`c4b434fe51fa481eb314f1714c1aebf6376033f308b63ec1fae0ee7e0f2c6ea7`。
- 验证清单 SHA256：`a43eaabae1f795edf5fb9472ca947734f150abd49a31e0e960e83e1acf447dec`。
- 完整统计：`data/BDD100K/splits/smoke_report.json`。
- 该子集仅用于工程冒烟，不得用于论文精度对比。

### BDD100K 全量基础统计

- 统计脚本：`scripts/analyze_bdd100k.py`。
- train：70,000 images、1,288,009 instances、small 710,395（55.15%）。
- val：10,000 images、185,526 instances、small 102,729（55.37%）。
- 总计：80,000 images、1,473,535 instances、small 813,124（55.18%）。
- 80,000 张图像均为 1280×720，空标签图像为 0。
- 统计过滤结果为 447 个退化框和 1 个重复框，与 YOLO 转换报告一致。
- 天气、场景、时段、遮挡、截断、类别图像数和完整 JSON：`results/dataset_stats/bdd100k/`。

## VisDrone2019-DET

- 数据源：Ultralytics GitHub Release 镜像；test-dev 慢链路部分使用旧 YOLOv5 Release 续传。
- 切换 test-dev 源前，对 1 MiB 重叠区间逐字节 `cmp`，结果一致。
- train ZIP SHA256：`86a77eba93137bfc16e4993860de9245b0675c0dba0d3ab98fb458699e256f84`。
- val ZIP SHA256：`abeea063037e5d20398837deb11084e652402a34ddf4f207bdf541a6f2a35ef9`。
- test-dev ZIP SHA256：`b28a94b06dfd9e36ce77ff8155fb82b9d2c030f198a76105933bd56c6ea6a68d`。
- test-dev ZIP 完整性：`unzip -t` 无错误。
- 图像与标签：6,471 train、548 val、1,610 test-dev。
- YOLO 标签树 SHA256：`137d120a0cfaf82ee85433354c75855a6ea1e2d49227938c82b6355df23fcdfb`。
- Ultralytics 全量扫描：8,629 images、8,629 labels、0 corrupt。
- 移除 4 个完全重复框。
- 原图尺寸复算：train 343,201实例（small 60.49%），val 38,759实例（small 68.59%），test-dev 75,102实例（small 67.69%）；详见`results/dataset_stats/visdrone/`。
- test-dev GT公开，冻结为最终一次性测试集；不得参与模型选择或调参。
- YOLO转换会跳过原标注`score=0`的ignored regions。内部Ultralytics配对指标可复现，但论文对外数字必须另用VisDrone官方评估工具处理ignored regions。
- Git worktree通过被忽略的`data/VisDrone -> ../../../data/VisDrone`符号链接访问主工作区唯一数据副本，避免复制3.7 GB数据。
- 已从原始ZIP恢复官方annotations：val 548份、树SHA256 `50d7697b...`；test-dev 1,610份、树SHA256 `f66f642d...`。恢复目录位于被Git忽略的数据树，不修改原ZIP。

类别顺序与实例数：

| ID | 类别 | 实例数 |
|---:|---|---:|
| 0 | pedestrian | 109,185 |
| 1 | people | 38,560 |
| 2 | bicycle | 13,069 |
| 3 | car | 187,004 |
| 4 | van | 32,702 |
| 5 | truck | 16,284 |
| 6 | tricycle | 6,387 |
| 7 | awning-tricycle | 4,377 |
| 8 | bus | 9,117 |
| 9 | motor | 40,377 |

准备脚本：`scripts/prepare_visdrone.py`。数据配置：`configs/data/visdrone.yaml`。

## 存储状态

- `data/BDD100K`：约 16 GB，包含原始 ZIP、解压图像、YOLO 标签和可追溯标注快照。
- `data/VisDrone`：约 3.7 GB，包含原始 ZIP、解压图像和 YOLO 标签。
- 当前工作盘剩余约 301 GB。

大文件均被项目根目录 `.gitignore` 排除，不应提交到代码仓库。
