# 学校论文模板接入清单

> **[已归档 2026-09-07]** 其图表/映射清单基于P11-P15材料，不含P18 ACR；取得学校模板、重写论文时按新目录结构重建清单。仅作历史审计保留。
>
> 用途：在不假定学校格式的前提下，把`THESIS_MANUSCRIPT_DRAFT.md`映射到正式论文源稿。

## 待取得的模板输入

- 学校官方Word或LaTeX模板及版本号。
- 论文中文、英文题名格式要求。
- 作者、学号、学院、专业、研究方向、导师和完成日期。
- 学位类型及是否要求单列工程应用、创新点或数据伦理说明。
- 摘要字数、关键词数量、目录层级和章节命名规则。
- 图题、表题、公式、单位、缩写和附录格式。
- 参考文献标准、引用样式及学校提供的BST/CSL文件。
- 查重、匿名评审和数据/代码公开要求。

取得这些输入前，不填写虚构个人信息，不固定章节号，不把“图X”替换为猜测编号，也不转换参考文献样式。

## 正文映射

| 整合稿内容 | 建议模板位置 | 主要来源 |
|---|---|---|
| 摘要、关键词 | 前置摘要 | `THESIS_MANUSCRIPT_DRAFT.md` |
| 研究背景与问题 | 绪论 | 整合稿及`PAPER_METHODS_AND_CAPTIONS.md` |
| 相关工作 | 相关工作/理论基础 | 整合稿及`references.bib` |
| 方法与实验体系 | 方法、实验设计 | 整合稿、两份冻结协议 |
| 实验结果 | 实验结果与分析 | 整合稿及`RESULTS.md` |
| 讨论、局限性 | 讨论 | 整合稿及`THESIS_CONTRIBUTION_ASSESSMENT.md` |
| 结论 | 总结与展望 | 整合稿 |
| 完整负结果矩阵 | 正文表格或附录 | `RESULTS.md`及分类实验报告 |
| 配置、哈希与复现命令 | 附录 | `PROJECT_STATUS.md`、manifest和脚本 |

章节编号以学校模板为准。正文映射时保留“官方P2不是本文原创”“BDD学习路由未超过启发式”“ScoutDet-RV未过强树5%门”“运行时无加速优势”四项边界。

## 图件清单

| 资产名 | 内容 | 推荐位置 |
|---|---|---|
| `bdd-size-ap-comparison` | BDD原生、P2和通道注意力尺寸AP | BDD基线结果 |
| `p2-repaired-small` | P2修复small漏检 | BDD误差分析 |
| `shared-small-miss` | 两模型共同small漏检 | BDD误差分析 |
| `p2-added-false-positive` | P2新增误检 | BDD误差分析 |
| `visdrone-training-curves` | 三seed训练曲线 | VisDrone稳健性 |
| `visdrone-multiseed-size-errorbars` | 三seed尺寸AP/AR误差条 | VisDrone稳健性 |
| `visdrone-val-test-size-gap` | val/test-dev尺寸差距 | 泛化讨论 |
| `visdrone-official-class-gap` | official-compatible分类别差距 | 泛化讨论 |

PNG用于Word编辑预览，PDF优先用于LaTeX或最终矢量排版。正式插图前从`results/paper_figures/manifest.json`和`results/bdd-error-analysis/manifest.json`核对来源与哈希。

## 关键表格

- BDD原生YOLO26n与官方P2完整200轮主表。
- BDD结构、损失、蒸馏和增强候选负结果矩阵。
- 全640、全960、随机、启发式、学习路由与Oracle比较。
- 集成运行时总时间、吞吐和峰值显存表。
- VisDrone Stage20模型选择表。
- VisDrone三seed均值与样本标准差表。
- VisDrone val与唯一一次test-dev对照表。
- VisDrone 640/960/25% Oracle上限表。
- ScoutDet-RV三seed、启发式、Ridge与HistGradientBoosting内部停止门表。

表内保留指标定义和训练预算，避免把Stage20与完整训练结果放在同一列而不注明预算。

## 引用与术语

- 引用键统一来自`docs/paper/references.bib`，不得手工新增无法核验的条目。
- `official-compatible`不得改写为“官方MATLAB评估”。
- `P2`写为官方YOLO26配置；`SDD`历史模块与SDD-YOLO论文名称须明确区分。
- Bootstrap写为验证图像采样不确定性，不写成训练seed置信区间。
- “未分出差异”不得改写为“等价”；“相对全640正收益”不得改写为“优于启发式”。
- P13写为假设开发上限，不写成可部署Oracle；P14只写train序列隔离holdout，不写成val或test外部验证。
- 没有Jetson、TensorRT或相应硬件实测时，不写“完成边缘部署”或“实时”。

## 导出前检查

1. 用学校模板完成章节号、图表号、公式号和交叉引用。
2. 将所有资产名替换为模板中的交叉引用，不保留“图X”占位。
3. 核对摘要、正文、结论中的关键数值完全一致。
4. 核对所有引用键均由模板工具成功解析。
5. 运行默认和强制外部证据论文审计。
6. 运行仓库布局审计、完整单元测试和`git diff --check`。
7. 检查最终PDF字体嵌入、图件清晰度、表格跨页和书签目录。
8. 人工复核禁止表述、test-dev封存边界和官方结构署名。

当前整合稿已完成内容层收口；学校模板和个人元数据是正式排版阶段唯一未提供的外部输入。
