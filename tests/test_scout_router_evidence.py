"""Regression checks for the frozen P14 internal stop decision."""

from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ScoutRouterEvidenceTest(unittest.TestCase):
    def test_internal_gate_stops_before_val(self) -> None:
        summary = json.loads((ROOT / "results/scout-router-screen/summary.json").read_text(encoding="utf-8"))
        self.assertFalse(summary["passed"])
        self.assertEqual(summary["status"], "stop before val")
        self.assertFalse(summary["checks"]["ensemble_relative_gain_at_least_5_percent"])
        self.assertTrue(summary["checks"]["all_seeds_not_below_best_non_neural"])
        self.assertLess(summary["ensemble_relative_utility_capture_percent"], 5.0)

    def test_all_inputs_are_train_only_and_three_seeds_are_present(self) -> None:
        summary = json.loads((ROOT / "results/scout-router-screen/summary.json").read_text(encoding="utf-8"))
        self.assertEqual(set(summary["seeds"]), {"0", "1", "2"})
        for item in summary["inputs"].values():
            report_path = ROOT / "results/scout-router-screen" / Path(item["report"]).name
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertIn("val and test-dev excluded", report["status"])
            self.assertEqual(report["groups"], {"optimize": 128, "selection": 32, "holdout": 48})


if __name__ == "__main__":
    unittest.main()
