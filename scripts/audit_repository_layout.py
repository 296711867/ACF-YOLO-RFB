"""Audit the documented repository layout and local Markdown links."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ROOT_MARKDOWN = {"README.md", "PROJECT_STATUS.md", "SESSION_HANDOFF.md", "NEXT_STAGE_PLAN.md", "服务器AI后续实验执行说明_优先版.md"}
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
    if actual_root_markdown != ROOT_MARKDOWN:
        failures.append(
            f"root Markdown mismatch: expected {sorted(ROOT_MARKDOWN)}, found {sorted(actual_root_markdown)}"
        )
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
