# 实验结果导航

`results/`同时保存机器证据和人读报告。机器目录保持原路径稳定；Markdown报告统一放入`reports/`。

## 报告分类

| 目录 | 内容 |
|---|---|
| `reports/baselines/` | Phase 0、原生YOLO26n与官方P2基线 |
| `reports/bdd-modules/` | SDD、DWF、SA-NWD、分辨率与通道注意力消融 |
| `reports/original-methods/` | 蒸馏、RDCI、PASCP、PGSR和学习路由 |
| `reports/visdrone/` | VisDrone协议、预检、Stage20、多seed和最终结果 |
| `reports/analysis/` | BDD配对Bootstrap与定性误差分析 |

当前最重要的报告是：

- `reports/baselines/phase1_official_p2_640_report.md`
- `reports/analysis/bdd_paired_bootstrap_report.md`
- `reports/analysis/router_paired_bootstrap_report.md`
- `reports/analysis/integrated_router_runtime_report.md`
- `reports/analysis/visdrone_resolution_oracle_report.md`
- `reports/analysis/p15_paper_closure_report.md`
- `reports/analysis/p16_repository_cleanup_report.md`
- `reports/original-methods/perspective_router_final_report.md`
- `reports/original-methods/pre_inference_utility_router_screen_report.md`
- `reports/original-methods/scout_detection_value_router_prep_report.md`
- `reports/original-methods/scout_detection_value_router_screen_report.md`
- `reports/visdrone/visdrone_multiseed_report.md`

## 机器证据

命名目录如`baseline/`、`p2/`、`rdci-stage20/`、`pascp-stage20/`、`perspective-router-final/`、`integrated-router-runtime/`、`visdrone-resolution-oracle/`和`visdrone-multiseed/`包含JSON、CSV、NPZ、checkpoint或manifest。`paper_figures/`和`bdd-error-analysis/`包含由冻结数据生成的图件。

这些目录没有按报告分类移动，因为脚本、manifest、哈希记录和论文审计依赖稳定路径。报告移动只改变文档导航，不改变实验数据来源。

## 大文件

`.gitignore`排除原始预测、训练run、权重和数据集。正式路由六份预测JSONL的规范副本保存在根项目的`results/perspective-router-final/`，不在每个新工作树重复复制。删除任何忽略文件前先查`docs/REPOSITORY_MAP.md`并核对SHA256。
