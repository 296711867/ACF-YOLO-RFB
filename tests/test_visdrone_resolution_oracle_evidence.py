"""Regression checks for the frozen P13 machine evidence."""

from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/visdrone-resolution-oracle"


class VisDroneResolutionOracleEvidenceTest(unittest.TestCase):
    def test_gate_is_passing_and_uses_val_only(self) -> None:
        gate = json.loads((RESULTS / "gate.json").read_text(encoding="utf-8"))
        oracle = json.loads((RESULTS / "oracle.json").read_text(encoding="utf-8"))
        self.assertTrue(gate["passed"])
        self.assertTrue(all(gate["checks"].values()))
        self.assertEqual(oracle["images"], 548)
        self.assertEqual(oracle["split"], "val")
        self.assertNotIn("test", oracle["baseline"].lower())
        self.assertNotIn("test", oracle["high_res"].lower())

    def test_exact_budget_and_frozen_metrics(self) -> None:
        gate = json.loads((RESULTS / "gate.json").read_text(encoding="utf-8"))
        oracle = json.loads((RESULTS / "oracle.json").read_text(encoding="utf-8"))
        self.assertEqual(oracle["budgets"]["0.25"]["routed_images"], 137)
        self.assertEqual(gate["metrics"]["baseline"]["small_ap"], 0.098479)
        self.assertEqual(gate["metrics"]["full960"]["small_ap"], 0.140552)
        self.assertEqual(gate["metrics"]["oracle25"]["small_ap"], 0.125729)


if __name__ == "__main__":
    unittest.main()
