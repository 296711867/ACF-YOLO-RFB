"""Tests for the frozen official-P2 versus RDCI Stage 20 matrix."""

from __future__ import annotations

import unittest
from pathlib import Path

from ultralytics.utils import YAML


ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "configs/train/bdd100k-rdci-r0-p2-control-640-e200-stop20-b32.yaml"
CANDIDATE = ROOT / "configs/train/bdd100k-rdci-r1-candidate-640-e200-stop20-b32.yaml"


class RDCIStage20ConfigTest(unittest.TestCase):
    """Prevent optimization and data drift between the official P2 control and RDCI."""

    def test_matrix_only_changes_model_and_name(self) -> None:
        """All non-treatment settings must remain exactly equal."""
        control = YAML.load(CONTROL)
        candidate = YAML.load(CANDIDATE)
        treatment_keys = {"model", "name"}
        self.assertEqual(
            {key: value for key, value in control.items() if key not in treatment_keys},
            {key: value for key, value in candidate.items() if key not in treatment_keys},
        )
        self.assertEqual(control["model"], "upstream/ultralytics/cfg/models/26/yolo26-p2.yaml")
        self.assertEqual(candidate["model"], "configs/models/yolo26n-p2-rdci.yaml")
        self.assertNotEqual(control["name"], candidate["name"])
        self.assertEqual(control["epochs"], 200)
        self.assertEqual(control["stage_stop_epoch"], 20)
        self.assertEqual(control["batch"], 32)
        self.assertEqual(control["nbs"], 64)
        self.assertFalse(control["exist_ok"])


if __name__ == "__main__":
    unittest.main()
