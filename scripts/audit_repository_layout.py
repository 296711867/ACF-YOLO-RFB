"""Audit the documented repository layout and local Markdown links."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ROOT_MARKDOWN = {"README.md", "PROJECT_STATUS.md"}
OPTIONAL_ROOT_MARKDOWN = {"AGENTS.md"}
INTERNAL_HANDOFF = "docs/internal/SESSION_HANDOFF.md"
ARCHIVED_STATUS_DOCS = ("NEXT_STAGE_PLAN.md", "SESSION_HANDOFF.md")
OBSOLETE_DOCUMENTS = {
    "YOLO26道路小目标检测_完整研究与服务器实现方案.md",
    "道路小目标检测硕士论文_从0到1实施规划.md",
}
LINK_PATTERN = re.compile(r"!?\[[^]]*]\(([^)]+)\)")


def markdown_files() -> list[Path]:
    """Return project Markdown files without auditing the frozen upstream snapshot."""
    return [path for path in ROOT.rglob("*.md") if "upstream" not in path.relative_to(ROOT).parts]


def audit() -> list[str]:
    """Return layout and local-link failures."""
    failures = []
    actual_root_markdown = {path.name for path in ROOT.glob("*.md")}
    unexpected_root_markdown = actual_root_markdown - ROOT_MARKDOWN - OPTIONAL_ROOT_MARKDOWN
    missing_root_markdown = ROOT_MARKDOWN - actual_root_markdown
    if missing_root_markdown:
        failures.append(f"root Markdown missing: {sorted(missing_root_markdown)}")
    if unexpected_root_markdown:
        failures.append(f"unexpected root Markdown: {sorted(unexpected_root_markdown)}")
    if not (ROOT / INTERNAL_HANDOFF).is_file():
        failures.append(f"internal session handoff missing: {INTERNAL_HANDOFF}")
    for name in ARCHIVED_STATUS_DOCS:
        if (ROOT / name).exists():
            failures.append(f"archived status document still at root: {name}")
    for name in OBSOLETE_DOCUMENTS:
        if (ROOT / name).exists():
            failures.append(f"obsolete document still exists: {name}")
    for path in markdown_files():
        text = path.read_text(encoding="utf-8")
        for raw_target in LINK_PATTERN.findall(text):
            target = raw_target.strip().strip("<>").split("#", 1)[0]
            if not target or "://" in target or target.startswith(("mailto:", "#")):
                continue
            resolved = (path.parent / target).resolve()
            if not resolved.exists():
                failures.append(f"broken local link: {path.relative_to(ROOT)} -> {raw_target}")
    return failures


def main() -> None:
    """Run the layout audit."""
    failures = audit()
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"Repository layout audit passed: {len(markdown_files())} Markdown files checked.")


if __name__ == "__main__":
    main()
