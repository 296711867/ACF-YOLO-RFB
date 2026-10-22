> **口径提示(2026-10-22)**:本导航中各文档若含"ACR 最终方法/封箱"等表述,均为 ACR 阶段历史口径;论文 v2 最终模型为 ACF-YOLO-RFB,变更对照见 [paper/THESIS_RESULTS.md](paper/THESIS_RESULTS.md) §5。

# 文档导航

文档按用途而非时间堆放。当前事实只在根目录`PROJECT_STATUS.md`维护；历史协议保留用于审计，不是待执行方案。

## 当前入口

| 文档 | 用途 |
|---|---|
| `../PROJECT_STATUS.md` | 当前结论、完整实验历史和风险，唯一状态源 |
| `../README.md` | 公开仓库入口：项目简介、核心结果、快速开始 |
| `REPOSITORY_MAP.md` | 代码、配置、工作树和大文件生命周期 |
| `internal/SESSION_HANDOFF.md` | 新会话环境、活动分支、可信结论和禁止事项（**内部**，不参与公开发布） |
| `OPEN_SOURCE_RELEASE.md` | 开源发布手册：公开/内部划分、脱敏清单、导出流程 |

`../NEXT_STAGE_PLAN.md`已于2026-09-07归档至`archive/plans/`（P18封箱后被取代）。

最近两次生命周期清理及保留边界见`../results/reports/analysis/p16_repository_cleanup_report.md`与`p18_repository_cleanup_report.md`。

## 论文材料

`paper/`只保存写作中的活文件；数字唯一权威是`RESULTS.md`：

- `paper/RESULTS.md`：冻结正式结果主表（唯一数字权威，含P18 ACR §1.10）。
- `paper/THESIS_MANUSCRIPT_DRAFT.md`：无编号完整论文整合稿（P15时期成稿，ACR章节待导师追认后并入，见其顶部写作基线声明）。
- `paper/references.bib`：已核验引用库。
- `paper/archive/`：五份被取代的旧稿（结果叙述稿、方法图注稿、SCI定位、贡献评估、模板清单），每份顶部有归档原因；其中贡献评估与SCI定位含内部决策，不参与公开发布。

## 协议

- `protocols/ACR_STAGE20_PROTOCOL.md`：P18B非对称容量重分配Stage20筛选协议（2026-08-28冻结；已通过并完成200轮闭环，见`../results/reports/bdd-modules/acr_stage20_report.md`）。
- `protocols/ACR_SEED1_PROTOCOL.md`、`protocols/ACR_SEED2_PROTOCOL.md`：ACR多seed复制协议（均已执行并通过跨seed门；P18实验线2026-09-07封箱）。
- `protocols/WIDEN_CONTROL_STAGE20_PROTOCOL.md`：P18C容量对齐对照（widen-only）协议（2026-09-01冻结并启动前修订为全200轮；已收口：宽度与重分配不可分）。
- `protocols/HAAR_DOWNSAMPLE_STAGE20_PROTOCOL.md`：P18A Haar小波下采样Stage20筛选协议（2026-08-27冻结；已执行并于08-28按门关闭，作为ACR容量假设的反证消融保留）。
- `protocols/PERSPECTIVE_ROUTER_FINAL_PROTOCOL.md`：已执行的正式10k路由协议。
- `protocols/INTEGRATED_RUNTIME_PROTOCOL.md`：已执行的500图集成运行时协议。
- `protocols/PRE_INFERENCE_UTILITY_ROUTER_PROTOCOL.md`：P12检测前效用排序路由的工程、开发与停止门。
- `protocols/VISDRONE_RESOLUTION_ORACLE_PROTOCOL.md`：P13冻结VisDrone 640/960与25% GT Oracle外部上限门。
- `protocols/SCOUT_DETECTION_VALUE_ROUTER_PROTOCOL.md`：P14 ScoutDet-RV集合路由、序列隔离、基线与四阶段停止门。
- `archive/protocols/`：蒸馏、RDCI、PASCP和PGSR失败路线的冻结研究协议。只用于复查原创边界、实验公平性和停止门，不得按其中“下一步”恢复实验。

## 候选方案

`plans/`保存候选方向记录，顶部§0为各方案最终状态（A已关闭、G=ACR已封箱为最终方法、B-F未启动）：

- `plans/MODULE_IMPROVEMENT_PROPOSALS.md`：拟P18频域/小波模块改进候选（2026-08-27记录）。剩余候选启动前须单独冻结预注册协议。

## 归档

- `archive/protocols/`：失败路线冻结协议（见上）。
- `archive/plans/NEXT_STAGE_PLAN.md`：P16时期下一阶段计划，P18封箱后归档。

## 已删除材料

2026-08-08删除两份早期总方案：

- `YOLO26道路小目标检测_完整研究与服务器实现方案.md`基于实验前的YOLO26-SDF假设，所列SDD/DWF/SA-NWD组合已全部被证据否定。
- `道路小目标检测硕士论文_从0到1实施规划.md`基于更早的YOLO11、ADGF和DRSR设想，与当前代码和研究问题不一致。

两者的仍有效内容已由`PROJECT_STATUS.md`、`REPOSITORY_MAP.md`、论文材料和冻结协议覆盖，因此不另建“旧文档”副本。
