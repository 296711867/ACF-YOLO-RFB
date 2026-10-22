# 实验结果导航

`results/`同时保存机器证据和人读报告。机器目录保持原路径稳定；Markdown报告统一放入`reports/`。

## 报告分类

| 目录 | 内容 |
|---|---|
| `reports/baselines/` | Phase 0、原生YOLO26n与官方P2基线 |
| `reports/bdd-modules/` | SDD、DWF、SA-NWD、分辨率、通道注意力、Haar下采样、ACR容量重分配与ACFR融合消融 |
| `reports/original-methods/` | 蒸馏、RDCI、PASCP、PGSR和学习路由 |
| `reports/visdrone/` | VisDrone协议、预检、Stage20、多seed和最终结果 |
| `reports/analysis/` | BDD配对Bootstrap、定性误差分析、ACR/ACFR可视化分析 |
| `reports/lightweight/` | P20第五章轻量化：逐层成本审计、单元筛选负结果、失败根因与方向分析 |
| `reports/external/` | 外部验证审计（P17 UAVDT收口审计） |

当前最重要的报告是：

- `reports/bdd-modules/acr_stage20_report.md`（P18 ACR完整证据链，已封箱）
- `reports/bdd-modules/acfr_stage20_report.md`（P19A ACFR完整证据链：三seed+组件消融+TRT收口）
- `reports/baselines/phase1_official_p2_640_report.md`
- `reports/lightweight/lightweight_unit_stage20_report.md`（P20单元筛选负结果终版）
- `reports/lightweight/unit_failure_analysis_and_directions.md`（失败根因+第五章A/B方向，导师材料）
- `reports/external/p17_uavdt_closure_audit.md`（P17只读审计：性能失败非工程失败）
- `reports/analysis/acr_acfr_visual_analysis.md`（ACR/ACFR补图筛选规则与解读）
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

命名目录如`baseline/`、`p2/`、`acr-stage20/`、`acfr-stage20/`、`lightweight-unit-screen/`、`lightweight-position-ablation/`、`rdci-stage20/`、`pascp-stage20/`、`perspective-router-final/`、`integrated-router-runtime/`、`visdrone-resolution-oracle/`和`visdrone-multiseed/`包含JSON、CSV、NPZ、checkpoint或manifest。`paper_figures/`、`paper-visuals/acr-acfr/`和`bdd-error-analysis/`包含由冻结数据生成的图件。

这些目录没有按报告分类移动，因为脚本、manifest、哈希记录和论文审计依赖稳定路径。报告移动只改变文档导航，不改变实验数据来源。

## 大文件

`.gitignore`排除原始预测、训练run、权重和数据集。正式路由六份预测JSONL的规范副本保存在根项目的`results/perspective-router-final/`，不在每个新工作树重复复制。删除任何忽略文件前先查`docs/REPOSITORY_MAP.md`并核对SHA256。
