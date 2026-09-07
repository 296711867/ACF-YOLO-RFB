# YOLO26 道路小目标检测研究工程

本仓库保存 BDD100K 与 VisDrone 道路小目标检测的可复现实验、预注册协议、负结果和论文证据。全部结论遵循"协议先冻结、门数值训练前定、正式结果只从冻结日志与机器证据生成"的纪律。

## 核心结果

**最终方法：ACR（非对称容量重分配）**——对官方 YOLO26-P2 头部做两处宽度调整（L19 P2 检测分支 32→64 通道、L28 P5 头分支 256→192 通道），backbone 不变，**纯 YAML 配置、零自定义代码**（`configs/models/yolo26n-p2-acr.yaml`）。

BDD100K 10k val、640 输入、COCO 尺寸口径（完整表格见 [`docs/paper/RESULTS.md`](docs/paper/RESULTS.md)§1.10）：

| 模型 | 参数 | all AP | small AP | small AR |
|---|---:|---:|---:|---:|
| 原生 YOLO26n-640 | 2,518,352 | 0.255413 | 0.088772 | — |
| 官方 YOLO26n-P2-640 | 2,662,400 | 0.268221 | 0.101480 | — |
| **ACR（本项目）** | **2,490,624** | **0.280471** | **0.110785** | 0.308011 |

- 同协议配对 200 轮：相对官方 P2 对照 all AP +4.15%、**small AP +7.28%**；1,000 次同图像配对 Bootstrap 95% 区间均严格大于 0。
- 三 seed 复现：small AP 相对对照 +7.28%/+5.93%/+5.24% 全部过预注册跨 seed 门；均值 0.109619 ± 0.001070。
- 参数较官方 P2 **净减 6.45%**，推理实测**快 4.24%**（5.844 vs 6.103 ms/图，RTX 3090、FP32、batch16）。
- 三方消融（对照/ACR/widen-only）判定：P2 加宽是 small 增益主导因素，非对称重分配以三者最低参数取得同等精度，定位为效率型贡献。
- 机制链：误差分析发现容量倒挂（small 目标由最薄的 32 通道分支负责）；Haar 小波无损下采样对照（small AP -0.89%）反证信息保存非瓶颈、表示容量才是。

**其他主线**：8,769 参数透视分带分辨率路由相对全 640 取得 all AP +4.31%、small AP +13.75%，但 small AP 未超过同预算粗检启发式（配对 Bootstrap 区间跨 0），按预注册门保留为正向候选而非已证明必要的方法；VisDrone 保留原生 YOLO26n 三 seed 结果，test-dev 只访问一次并永久封存。**负结果边界**（SDD、DWF、SA-NWD、通道注意力、蒸馏、RDCI、PASCP、裁块、Haar 下采样等）见 [`docs/paper/RESULTS.md`](docs/paper/RESULTS.md)§4。

## 目录

```text
configs/        数据、模型与训练配置（ACR 模型在 configs/models/yolo26n-p2-acr.yaml）
docs/           论文材料、冻结协议、计划与归档（见 docs/README.md）
environment/    环境、硬件与数据审计记录
results/        机器证据与分类实验报告（见 results/README.md）
scripts/        训练、评估、统计、审计与导出工具
sdf/            自研模块源码（均为已关闭路线，保留供负结果复查）
tests/          工程、指标与论文声明回归测试
upstream/       冻结的 Ultralytics 8.4.104 源码快照（AGPL-3.0）
```

`runs/`、`data/`、权重和大体积原始预测由 Git 忽略。它们不能仅凭文件夹名称删除，须先按 [`docs/REPOSITORY_MAP.md`](docs/REPOSITORY_MAP.md) 核对是否已有哈希与规范副本。

## 环境与验证

```bash
conda activate yolo26-sdf
python -m pip install --no-build-isolation --no-deps -e .
PYTHONPATH=upstream:. python scripts/verify_setup.py
PYTHONPATH=upstream:. python -m unittest discover -s tests -v
PYTHONPATH=upstream:. python scripts/audit_paper_claims.py
PYTHONPATH=upstream:. python scripts/audit_repository_layout.py
```

正式训练只能通过 `scripts/run_experiment.py` 启动，异常续训只能使用 `scripts/resume_experiment.py`。启动器会校验干净工作树、配置、数据、checkpoint 和输出目录，避免覆盖既有证据。

## 文档入口

| 需要什么 | 去哪里 |
|---|---|
| 正式结果主表（唯一数字权威） | [`docs/paper/RESULTS.md`](docs/paper/RESULTS.md) |
| 论文活稿与文献库 | `docs/paper/THESIS_MANUSCRIPT_DRAFT.md`、`docs/paper/references.bib` |
| 冻结预注册协议（10 份） | [`docs/protocols/`](docs/protocols) |
| 实验报告导航 | [`results/README.md`](results/README.md) |
| 仓库工件生命周期地图 | [`docs/REPOSITORY_MAP.md`](docs/REPOSITORY_MAP.md) |
| 完整实验历史与当前状态 | [`PROJECT_STATUS.md`](PROJECT_STATUS.md) |
| 开源发布手册 | [`docs/OPEN_SOURCE_RELEASE.md`](docs/OPEN_SOURCE_RELEASE.md) |

本仓库是研究工程的公开发布快照（由 `scripts/build_public_release.py` 从内部审计库导出，单提交历史）；完整实验过程历史保存在内部审计仓库中。

## 许可与引用

本仓库以 **AGPL-3.0** 发布（见 [`LICENSE`](LICENSE)）；`upstream/` 为修改版 Ultralytics 8.4.104 快照，遵循其原始 AGPL-3.0 许可与版权声明。

公开仓库地址：`https://github.com/296711867/yolo26-acr`（由 `scripts/build_public_release.py` 从内部审计库导出维护，请勿在公开副本上直接提交）。论文引用条目待发表后补充。

## 研究纪律

1. 正式结果只从冻结日志、JSON/CSV/JSONL 和可复现脚本生成。
2. 失败路线不组合、不恢复长训，也不从同一验证集继续调参。
3. 官方 P2 必须明确写为官方结构；GT Oracle 只能表示上限。
4. Bootstrap 只表示验证样本不确定性，不能替代训练多 seed。
5. VisDrone test-dev 不得再次访问，BDD 正式 val 不得再用于路由选型。
