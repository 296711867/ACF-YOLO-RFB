"""Regression tests for the P27 Inner-IoU and MPDIoU bbox variants."""

from __future__ import annotations

import math
import unittest

import torch

from sdf.losses import inner_bbox_iou, mpdiou
from ultralytics.utils.metrics import bbox_iou


class TestInnerBboxIoU(unittest.TestCase):
    def test_ratio_one_equals_ciou(self):
        pred = torch.tensor([[10.0, 10.0, 20.0, 20.0]])
        target = torch.tensor([[10.5, 10.5, 20.5, 20.5]])
        inner = inner_bbox_iou(pred, target, 1.0)
        ciou = bbox_iou(pred, target, xywh=False, CIoU=True)
        self.assertTrue(torch.allclose(inner, ciou, atol=1e-6))

    def test_identical_boxes_upper_bound(self):
        box = torch.tensor([[3.0, 4.0, 30.0, 24.0]])
        value = inner_bbox_iou(box, box, 0.8)
        # Same auxiliary boxes: plain IoU is 1 and penalty terms vanish.
        self.assertTrue(torch.allclose(value, torch.ones_like(value), atol=1e-4))

    def test_gradient_flows(self):
        pred = torch.tensor([[10.0, 10.0, 20.0, 20.0]], requires_grad=True)
        target = torch.tensor([[11.0, 11.0, 21.0, 21.0]])
        loss = (1.0 - inner_bbox_iou(pred, target, 0.8)).sum()
        loss.backward()
        self.assertIsNotNone(pred.grad)
        self.assertGreater(pred.grad.abs().sum().item(), 0.0)

    def test_batch_shape(self):
        pred = torch.rand(64, 4) * 10 + torch.tensor([0, 0, 10, 10])
        target = torch.rand(64, 4) * 10 + torch.tensor([0, 0, 10, 10])
        pred = torch.cat((pred[..., :2], pred[..., :2] + pred[..., 2:] / 2 + 1), dim=-1)
        target = torch.cat((target[..., :2], target[..., :2] + target[..., 2:] / 2 + 1), dim=-1)
        out = inner_bbox_iou(pred, target, 0.8)
        self.assertEqual(out.shape, (64, 1))


class TestMPDIoU(unittest.TestCase):
    def test_identical_boxes_equals_one(self):
        box = torch.tensor([[5.0, 6.0, 25.0, 30.0]])
        value = mpdiou(box, box)
        self.assertTrue(torch.allclose(value, torch.ones_like(value), atol=1e-4))

    def test_disjoint_boxes_below_zero(self):
        pred = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        target = torch.tensor([[5.0, 5.0, 6.0, 6.0]])
        value = mpdiou(pred, target)
        self.assertLess(value.item(), 0.0)

    def test_zero_offset_shift(self):
        # Pure translation by (dx, dy): IoU drops and point penalties are exact.
        box = torch.tensor([[0.0, 0.0, 4.0, 2.0]])
        shifted = torch.tensor([[1.0, 0.5, 5.0, 2.5]])
        inter = 3.0 * 1.5
        union = 8.0 + 8.0 - inter
        iou = inter / union
        cw = 5.0
        ch = 2.5
        expected = iou - (2 * (1.0**2 + 0.5**2)) / (cw**2 + ch**2 + 1e-7)
        value = mpdiou(box, shifted)
        self.assertAlmostEqual(value.item(), float(expected), places=5)

    def test_gradient_flows(self):
        pred = torch.tensor([[10.0, 10.0, 20.0, 20.0]], requires_grad=True)
        target = torch.tensor([[11.0, 11.0, 21.0, 21.0]])
        loss = (1.0 - mpdiou(pred, target)).sum()
        loss.backward()
        self.assertIsNotNone(pred.grad)
        self.assertGreater(pred.grad.abs().sum().item(), 0.0)


class TestWiring(unittest.TestCase):
    def test_variant_validation(self):
        from ultralytics.utils.loss import BboxLoss

        with self.assertRaises(ValueError):
            BboxLoss(reg_max=16, box_iou_variant="bogus")


if __name__ == "__main__":
    unittest.main()
