# 开源发布手册

> 建立日期：2026-09-07。本文是公开发布的唯一操作手册：划分规则、脱敏边界、License 决定、导出流程与发布前检查。
> 导出工具：`scripts/build_public_release.py`（纯标准库，无需训练环境）。

## 1. 发布范围与理由

**完整研究仓库**：全部冻结协议（`docs/protocols/`）、负结果报告与机器证据（`results/`）、已关闭路线源码（`sdf/`）、审计脚本与测试。论文定位即"预算受控、强基线、开放负结果的实证研究"，负结果证据链是核心贡献而非包袱。

不进入公开仓库的内容由导出脚本自动排除（git 未跟踪 + 显式排除清单），无需人工判断。

## 2. 公开/内部划分

| 类别 | 内容 | 处理 |
|---|---|---|
| 自动排除 | `runs/`、`data/`、`weights/`、全部 `.predictions.jsonl`、缓存、checkpoint | git 未跟踪，导出只取 `git ls-files` |
| 显式排除 | `docs/internal/`（会话交接、服务器路径、GPU 调度） | `PUBLIC_EXCLUSIONS` 清单 |
| 显式排除 | `docs/paper/archive/THESIS_CONTRIBUTION_ASSESSMENT.md` | 纯内部评审（毕业要求研判、决策分支） |
| 公开保留 | 其余归档论文旧稿（3 份） | 顶部有归档头；`test_thesis_manuscript` 依赖其中两份，不能排除 |
| 公开保留 | 协议/报告中的"导师追认""用户授权"等过程记录 | 属于研究纪律表述，无个人信息；导出时列为 review 提示人工确认 |
| README 补丁 | 内部 git 用法段落、docs/internal 目录行 | 导出时自动替换为公开快照说明 |

## 3. License 决定

**整仓 AGPL-3.0**（根 `LICENSE`）。理由：`upstream/` 是修改版 Ultralytics 8.4.104 快照（`tasks.py` 有本项目模块注册），本身为 AGPL-3.0；自有代码与其构成同一分发作品，最稳妥做法是同许可证发布并保留上游版权声明（`upstream/LICENSE` 及包内声明原样保留）。若未来希望自有代码改用 MIT 等宽松许可证，需先将 `upstream/` 改为 pip 依赖并移除源码快照——这会牺牲封闭路线的复现性，不建议在论文发表前做。

## 4. 历史策略

公开仓库使用**全新单提交历史**。理由：约 140 个证据文件内嵌服务器绝对路径（约 24 万行），脱敏只能作用于当前文件内容，全量历史无法干净脱敏；内部审计库（`.project.git`，148+ 提交）完整保留研究过程。导出提交信息会记录源内部 commit 哈希，保证可追溯。

## 5. 脱敏清单（导出脚本内置）

文本文件按序替换（`scripts/build_public_release.py` 的 `PATH_SCRUB_RULES`，长前缀优先）：

1. 内部仓库根路径（`/home/<内部用户名>/code/github-project/<内部项目名>/yolo`）→ `<REPO_ROOT>`
2. 内部项目父路径 → `<PROJECTS_ROOT>`
3. 内部 conda 安装路径 → `<CONDA_HOME>`
4. 内部用户主目录前缀 → `<HOME>`
5. 兜底：独立出现的内部用户名 token（如 hostname 字段）→ `<HOST_USER>`

具体字面值以脚本常量为准（`_INTERNAL_USER`/`_INTERNAL_PROJECT` 刻意以分片书写，使公开副本中的脚本本身不含完整标识）。

门禁（对导出后**每个文件**做二进制安全扫描）：

- **硬失败**（构建终止）：内部用户名 token、内部项目名 token（`HARD_FAIL_PATTERNS`）
- **review 提示**（列出文件名，人工确认后再发布）：`/home/`、`miniconda3`、`导师`、`授权启动`、`用户决定`

已知会触发 review 提示的合法位置：各协议与 `docs/paper/RESULTS.md` 的版本状态说明（"导师追认创新口径前暂不进入论文正文主表"）、`environment/`（conda 镜像 URL、硬件记录）。逐条确认没有新增姓名/路径即可。

## 6. 发布流程

```bash
# 1. 确认内部仓库干净、审计通过
git --git-dir=.project.git --work-tree=. status --short          # 应为空
PYTHONPATH=upstream:. python scripts/audit_repository_layout.py
PYTHONPATH=upstream:. python scripts/audit_paper_claims.py

# 2. 导出（输出到 public_release/，已 gitignore）
python scripts/build_public_release.py --dry-run                 # 先看规模
python scripts/build_public_release.py

# 3. 门禁通过后人工抽查（见 §7），然后创建 GitHub 仓库并推送
cd public_release
git remote add origin <新仓库URL>
git push -u origin main
```

注意：门禁失败时脚本保留 `public_release/` 便于排查，修复后删除该目录重跑；不要手工修补导出副本后继续（重新导出保证可复现）。

## 7. 发布前人工抽查清单

- [ ] `public_release/_BUILD_REPORT.md` 的文件数、脱敏数、review 列表符合预期
- [ ] 抽查 `results/acr-stage20/full200_summary.json`：路径已变为 `<REPO_ROOT>`，数字未被改动
- [ ] `README.md`：内部 git 段落已替换、无 `docs/internal` 引用、结果表数字与 `docs/paper/RESULTS.md`§1.10 一致
- [ ] `LICENSE` 在根目录且 `upstream/LICENSE` 原样保留
- [ ] review 提示文件逐条过目：只有过程性表述，无姓名、无新路径、无 IP
- [ ] `git -C public_release log --oneline` 恰好 1 个提交；`ls public_release/docs/internal` 不存在
- [ ] README 中"引用"占位是否要换成正式 BibTeX（论文录用后）
- [ ] GitHub 仓库描述、topics（object-detection、small-object、YOLO26、BDD100K、VisDrone）、可见性设置

## 8. 发布后维护

- 论文录用后：更新 README 引用块 → 内部仓库提交 → 重新导出推送（导出是确定性流程，可重复执行）
- 发现敏感残留：内部仓库修 `PUBLIC_EXCLUSIONS`/替换规则 → 删除远端历史有问题的提交（全新历史策略下直接重新导出强推即可）
- 不在公开仓库直接提交；公开副本始终由脚本从内部审计库生成
