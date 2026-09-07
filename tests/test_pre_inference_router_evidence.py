"""Evidence checks for the stopped P12 development screen."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/pre-inference-router-screen"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PreInferenceRouterEvidenceTest(unittest.TestCase):
    """Keep the negative result and preregistered stop decision traceable."""

    def test_seed0_failure_and_file_hashes_are_frozen(self) -> None:
        summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
        seed0 = json.loads((RESULTS / "seed0.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "stopped at seed0 development gate")
        self.assertFalse(summary["formal_bdd_val_accessed"])
        self.assertFalse(summary["visdrone_test_dev_accessed"])
        self.assertFalse(summary["gate"]["seed0_exceeds_best_image_baseline"])
        self.assertLess(
            seed0["dev_metrics"]["utility_capture"],
            seed0["dev_baselines"][seed0["best_dev_image_baseline"]]["utility_capture"],
        )
        self.assertEqual(sha256(RESULTS / "seed0.json"), summary["seed0"]["report_sha256"])
        self.assertEqual(
            sha256(RESULTS / "seed0.dev-scores.jsonl"),
            summary["seed0"]["scores_sha256"],
        )

    def test_report_preserves_the_stop_boundary(self) -> None:
        report = (ROOT / "results/reports/original-methods/pre_inference_utility_router_screen_report.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("不运行seed1或seed2", report)
        self.assertIn("不访问BDD正式val", report)
        self.assertIn("相对低11.34%", report)


if __name__ == "__main__":
    unittest.main()
