"""Tests for frozen VisDrone multi-seed replication configs."""

from __future__ import annotations

import unittest
from pathlib import Path

from ultralytics.utils import YAML


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs" / "train"


class VisDroneSeedConfigTest(unittest.TestCase):
    """Require seed replications to preserve the seed0 training protocol."""

    def test_seed_configs_only_change_seed_name_and_stage_pause(self) -> None:
        base = YAML.load(CONFIGS / "visdrone-yolo26n-640-e100-stage20-b16.yaml")
        base.pop("stage_stop_epoch")
        for seed in (1, 2):
            candidate = YAML.load(CONFIGS / f"visdrone-yolo26n-640-e100-b16-seed{seed}.yaml")
            expected = {**base, "seed": seed, "name": f"visdrone-yolo26n-640-e100-b16-seed{seed}"}
            self.assertEqual(candidate, expected)

    def test_seed_outputs_are_distinct(self) -> None:
        run_dirs = []
        for seed in (1, 2):
            config = YAML.load(CONFIGS / f"visdrone-yolo26n-640-e100-b16-seed{seed}.yaml")
            run_dirs.append(ROOT / config["project"] / config["name"])
        self.assertEqual(len(set(run_dirs)), 2)


if __name__ == "__main__":
    unittest.main()
