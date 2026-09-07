"""Tests for the frozen P2-to-P3 Stage 20 experiment matrix."""

from __future__ import annotations

import unittest
from pathlib import Path

from ultralytics.utils import YAML


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    "d0": ROOT / "configs/train/bdd100k-p2p3-d0-control-640-e200-stop20-b16.yaml",
    "d1": ROOT / "configs/train/bdd100k-p2p3-d1-common-640-e200-stop20-b16.yaml",
    "d2": ROOT / "configs/train/bdd100k-p2p3-d2-small-640-e200-stop20-b16.yaml",
    "d3": ROOT / "configs/train/bdd100k-p2p3-d3-consensus-640-e200-stop20-b16.yaml",
}


class P2P3Stage20ConfigTest(unittest.TestCase):
    """Prevent protocol drift across D0-D3."""

    def test_matrix_only_changes_distillation_treatment_and_name(self) -> None:
        """All optimization, data, initialization, and stage settings must remain identical."""
        configs = {name: YAML.load(path) for name, path in CONFIGS.items()}
        common_keys = {
            "task",
            "mode",
            "model",
            "pretrained",
            "data",
            "imgsz",
            "epochs",
            "stage_stop_epoch",
            "batch",
            "nbs",
            "device",
            "workers",
            "optimizer",
            "seed",
            "deterministic",
            "amp",
            "patience",
            "project",
            "exist_ok",
            "save",
            "plots",
            "verbose",
        }
        baseline = {key: configs["d0"][key] for key in common_keys}
        for name, config in configs.items():
            with self.subTest(group=name):
                self.assertEqual({key: config[key] for key in common_keys}, baseline)
                self.assertFalse(config["exist_ok"])
                self.assertEqual(config["epochs"], 200)
                self.assertEqual(config["stage_stop_epoch"], 20)
                self.assertEqual(config["batch"], 16)
                self.assertEqual(config["nbs"], 64)

        self.assertNotIn("distill_model", configs["d0"])
        self.assertNotIn("distill_mode", configs["d0"])
        expected_modes = {"d1": "p2p3_common", "d2": "p2p3_small", "d3": "p2p3_consensus"}
        teacher = configs["d1"]["distill_model"]
        for name, mode in expected_modes.items():
            self.assertEqual(configs[name]["distill_model"], teacher)
            self.assertEqual(configs[name]["distill_mode"], mode)
            self.assertEqual(configs[name]["dis"], 6.0)
        self.assertEqual(len({config["name"] for config in configs.values()}), 4)


if __name__ == "__main__":
    unittest.main()
