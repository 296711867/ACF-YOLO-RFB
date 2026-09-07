"""Tests for damped inverse-frequency detection class weights."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from ultralytics import YOLO
from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics.models.yolo.detect.train import DetectionTrainer


VISDRONE_COUNTS = np.asarray([79335, 27059, 10480, 144866, 24956, 12875, 4812, 3246, 5926, 29646])


class DetectionClassWeightsTest(unittest.TestCase):
    """Verify the frozen VisDrone class-weight behavior."""

    def test_quarter_power_weights_are_damped_and_mean_normalized(self) -> None:
        trainer = SimpleNamespace(args=SimpleNamespace(cls_pw=0.25))
        weights = DetectionTrainer.compute_class_weights(trainer, VISDRONE_COUNTS)
        weights /= weights.mean()

        self.assertAlmostEqual(float(weights.mean()), 1.0, places=6)
        self.assertAlmostEqual(float(weights.max() / weights.min()), 2.584666, places=5)
        self.assertEqual(int(weights.argmax()), 7)  # awning-tricycle
        self.assertEqual(int(weights.argmin()), 3)  # car

    def test_e2e_branches_receive_identical_class_weights(self) -> None:
        model = YOLO("upstream/ultralytics/cfg/models/26/yolo26.yaml").model
        expected = torch.linspace(0.5, 1.4, 10)
        model.args = SimpleNamespace(**DEFAULT_CFG_DICT)
        model.class_weights = expected
        criterion = model.init_criterion()

        self.assertTrue(torch.equal(criterion.one2many.class_weights.flatten().cpu(), expected))
        self.assertTrue(torch.equal(criterion.one2one.class_weights.flatten().cpu(), expected))


if __name__ == "__main__":
    unittest.main()
