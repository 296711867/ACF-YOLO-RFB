"""P30 regression tests: D-InterpIoU and PIoU2 loss variants on the L0+RFB baseline.

Faithfulness is enforced against the verbatim upstream reference implementations
stored in tests/fixtures (bit-exact output equality), per the P30 protocol §3.
"""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import torch

from sdf.losses import d_interpiou, piou2


def _load_reference(module_name: str):
    """Load a verbatim upstream reference implementation from tests/fixtures."""
    fixture = Path(__file__).parent / "fixtures" / module_name
    spec = importlib.util.spec_from_file_location(module_name.replace(".py", ""), fixture)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestDInterpIoU(unittest.TestCase):
    def test_matches_reference_bit_exact(self):
        reference = _load_reference("interpiou_reference.py")
        torch.manual_seed(30)
        pred = torch.rand(64, 4) * 100
        pred[:, 2:] = pred[:, 2:] + 5
        target = torch.rand(64, 4) * 100
        target[:, 2:] = target[:, 2:] + 5
        ours = d_interpiou(pred, target)
        ref = reference.d_interpiou(pred, target, xywh=False)
        self.assertTrue(torch.equal(ours, ref))

    def test_gradient_flows_in_nonoverlapping_case(self):
        pred = torch.tensor([[10.0, 10.0, 12.0, 12.0]])
        target = torch.tensor([[50.0, 50.0, 52.0, 52.0]])
        x = pred.clone().requires_grad_(True)
        d_interpiou(x, target).sum().backward()
        self.assertGreater(x.grad.abs().sum().item(), 0.0)

    def test_identical_boxes_max_similarity(self):
        box = torch.tensor([[10.0, 10.0, 20.0, 20.0]])
        value = d_interpiou(box, box.clone())
        self.assertAlmostEqual(value.item(), 1.0, places=5)


class TestPIoU2(unittest.TestCase):
    def test_matches_reference_bit_exact(self):
        reference = _load_reference("piou_reference.py")
        torch.manual_seed(31)
        pred = torch.rand(64, 4) * 100
        pred[:, 2:] = pred[:, 2:] + 5
        target = torch.rand(64, 4) * 100
        target[:, 2:] = target[:, 2:] + 5
        ours = piou2(pred, target)
        ref_loss = reference.piou(pred, target, xywh=False, PIoU2=True, Lambda=1.3)
        self.assertTrue(torch.equal(ours, 1.0 - ref_loss))

    def test_gradient_flows(self):
        pred = torch.tensor([[10.0, 10.0, 14.0, 14.0]])
        target = torch.tensor([[11.0, 11.0, 21.0, 21.0]])
        x = pred.clone().requires_grad_(True)
        piou2(x, target).sum().backward()
        self.assertGreater(x.grad.abs().sum().item(), 0.0)

    def test_identical_boxes_zero_loss(self):
        box = torch.tensor([[10.0, 10.0, 20.0, 20.0]])
        value = piou2(box, box.clone())
        self.assertAlmostEqual(1.0 - value.item(), 0.0, places=5)


class TestWiring(unittest.TestCase):
    def test_variants_accepted_by_bbox_loss(self):
        from ultralytics.utils.loss import BboxLoss

        for variant in ("d_interpiou", "piou2"):
            module = BboxLoss(reg_max=1, box_iou_variant=variant)
            self.assertEqual(module.box_iou_variant, variant)
        with self.assertRaises(ValueError):
            BboxLoss(reg_max=1, box_iou_variant="bogus")


if __name__ == "__main__":
    unittest.main()
