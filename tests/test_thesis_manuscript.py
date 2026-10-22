"""Tests for the template-neutral thesis manuscript package."""

from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "docs/paper/THESIS_MANUSCRIPT_DRAFT.md"
CHECKLIST = ROOT / "docs/paper/THESIS_TEMPLATE_INTEGRATION_CHECKLIST.md"
SCI_POSITIONING = ROOT / "docs/paper/SCI_SUBMISSION_POSITIONING.md"


class ThesisManuscriptTest(unittest.TestCase):
    """Keep the integrated draft complete without inventing template metadata."""

    def test_required_sections_and_claim_boundaries_are_present(self) -> None:
        text = MANUSCRIPT.read_text(encoding="utf-8")
        for heading in (
            "## 摘要",
            "## 关键词",
            "## Abstract",
            "## Keywords",
            "## 研究背景与问题",
            "## 相关工作",
            "## 方法与实验体系",
            "## 实验结果",
            "## 讨论",
            "## 结论",
        ):
            self.assertIn(heading, text)
        self.assertIn("不能证明学习头优于启发式", text)
        self.assertIn("不能表述为加速", text)
        self.assertIn("不把P2命名为本文原创模块", text)
        self.assertIn("相对强树基线仅提高3.35%", text)
        self.assertIn("不称为未见测试集验证", text)
        self.assertNotIn("图X", text)

    def test_template_checklist_preserves_unresolved_external_inputs(self) -> None:
        text = CHECKLIST.read_text(encoding="utf-8")
        self.assertIn("学校官方Word或LaTeX模板", text)
        self.assertIn("不填写虚构个人信息", text)
        self.assertIn("章节编号以学校模板为准", text)
        self.assertIn("git diff --check", text)

    def test_sci_positioning_uses_empirical_not_sota_claims(self) -> None:
        text = SCI_POSITIONING.read_text(encoding="utf-8")
        self.assertIn("预算受控实证研究", text)
        self.assertIn("不声称加速", text)
        self.assertIn("HistGradientBoosting", text)
        self.assertNotIn("全面优于现有方法", text)


if __name__ == "__main__":
    unittest.main()
