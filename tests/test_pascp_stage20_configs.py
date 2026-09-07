"""Tests for the frozen official-P2 versus PASCP Stage 20 matrix."""

from __future__ import annotations

import unittest
from pathlib import Path

from ultralytics.utils import YAML


ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "configs/train/bdd100k-pascp-r0-p2-control-640-e200-stop20-b32.yaml"
CANDIDATE = ROOT / "configs/train/bdd100k-pascp-r1-candidate-640-e200-stop20-b32.yaml"


class PASCPStage20ConfigTest(unittest.TestCase):
    """Prevent model, optimization, and augmentation drift outside the treatment probability."""

    def test_matrix_only_changes_probability_and_name(self) -> None:
        """All settings except PASCP activation and run name must remain equal."""
        control = YAML.load(CONTROL)
        candidate = YAML.load(CANDIDATE)
        treatment_keys = {"pascp", "name"}
        self.assertEqual(
            {key: value for key, value in control.items() if key not in treatment_keys},
            {key: value for key, value in candidate.items() if key not in treatment_keys},
        )
        self.assertEqual(control["model"], "upstream/ultralytics/cfg/models/26/yolo26-p2.yaml")
        self.assertEqual(candidate["model"], control["model"])
        self.assertEqual(control["pascp"], 0.0)
        self.assertEqual(candidate["pascp"], 0.5)
        self.assertEqual(control["epochs"], 200)
        self.assertEqual(control["stage_stop_epoch"], 20)
        self.assertEqual(control["batch"], 32)
        self.assertEqual(control["nbs"], 64)
        self.assertFalse(control["exist_ok"])


if __name__ == "__main__":
    unittest.main()
