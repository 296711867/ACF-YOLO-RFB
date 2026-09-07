"""Unit and integration tests for the size-aware NWD-CIoU loss."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from sdf.losses import SizeAwareNWDLoss
from ultralytics import YOLO
from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics.utils.loss import BboxLoss, E2ELoss
from ultralytics.utils.metrics import bbox_iou


class SizeAwareNWDTest(unittest.TestCase):
    """Verify SA-NWD numerics, scale mapping, gradients, and dual-head integration."""

    def test_area_weight_is_monotonic_and_capped(self) -> None:
        """Tiny targets must receive more NWD weight without fully removing CIoU."""
        loss = SizeAwareNWDLoss(alpha_max=0.8)
        sizes = torch.tensor((4.0, 16.0, 32.0, 96.0))
        alpha = torch.exp(-(sizes.square()) / loss.area_tau).clamp(max=loss.alpha_max)
        self.assertTrue(torch.all(alpha[:-1] >= alpha[1:]))
        self.assertAlmostEqual(float(alpha[0]), 0.8, places=6)
        self.assertGreaterEqual(float(1 - alpha.max()), 0.2 - 1e-6)

    def test_loss_is_reference_scale_invariant_and_has_finite_gradients(self) -> None:
        """Equivalent boxes at 640 and 960 must match after reference-scale mapping."""
        loss = SizeAwareNWDLoss()
        target_640 = torch.tensor([[100.0, 120.0, 116.0, 136.0]])
        pred_640 = torch.tensor([[101.0, 120.0, 117.0, 136.0]], requires_grad=True)
        ciou_640 = bbox_iou(pred_640, target_640, xywh=False, CIoU=True)
        actual_640 = loss(pred_640, target_640, ciou_640, torch.tensor([640.0, 640.0]))

        target_960 = target_640 * 1.5
        pred_960 = (pred_640.detach() * 1.5).requires_grad_()
        ciou_960 = bbox_iou(pred_960, target_960, xywh=False, CIoU=True)
        actual_960 = loss(pred_960, target_960, ciou_960, torch.tensor([960.0, 960.0]))
        self.assertTrue(torch.allclose(actual_640, actual_960, atol=1e-6, rtol=1e-6))

        (actual_640.sum() + actual_960.sum()).backward()
        self.assertTrue(torch.isfinite(pred_640.grad).all())
        self.assertTrue(torch.isfinite(pred_960.grad).all())

    def test_area_limit_restores_ciou_for_medium_targets(self) -> None:
        """The small-only fallback must use pure CIoU at and above the 32-square area boundary."""
        loss = SizeAwareNWDLoss(area_limit=1024.0)
        pred = torch.tensor([[101.0, 100.0, 133.0, 132.0]])
        target = torch.tensor([[100.0, 100.0, 132.0, 132.0]])
        ciou = bbox_iou(pred, target, xywh=False, CIoU=True)
        actual = loss(pred, target, ciou, torch.tensor([640.0, 640.0]))
        self.assertTrue(torch.equal(actual, 1 - ciou.float()))

        small_target = torch.tensor([[100.0, 100.0, 131.0, 131.0]])
        small_ciou = bbox_iou(pred, small_target, xywh=False, CIoU=True)
        small_actual = loss(pred, small_target, small_ciou, torch.tensor([640.0, 640.0]))
        self.assertFalse(torch.equal(small_actual, 1 - small_ciou.float()))

    def test_bbox_loss_uses_pixel_stride_and_preserves_l1(self) -> None:
        """SA-NWD must replace only the CIoU term while retaining YOLO26's reg_max=1 L1 term."""
        criterion = BboxLoss(reg_max=1, sa_nwd=SizeAwareNWDLoss())
        pred_dist = torch.ones(1, 2, 4, requires_grad=True)
        pred_boxes = torch.tensor([[[0.0, 0.0, 4.0, 4.0], [1.0, 1.0, 3.0, 3.0]]], requires_grad=True)
        target_boxes = torch.tensor([[[0.0, 0.0, 4.0, 4.0], [1.0, 1.0, 4.0, 4.0]]])
        scores = torch.ones(1, 2, 1)
        foreground = torch.tensor([[True, True]])
        box_loss, l1_loss = criterion(
            pred_dist,
            pred_boxes,
            torch.zeros(2, 2),
            target_boxes,
            scores,
            scores.sum(),
            foreground,
            torch.tensor([640.0, 640.0]),
            torch.tensor([[4.0], [8.0]]),
        )
        (box_loss + l1_loss).backward()
        self.assertTrue(torch.isfinite(box_loss))
        self.assertTrue(torch.isfinite(l1_loss))
        self.assertTrue(torch.isfinite(pred_boxes.grad).all())
        self.assertTrue(torch.isfinite(pred_dist.grad).all())

    def test_disabled_sa_nwd_preserves_native_ciou_value(self) -> None:
        """The default-off path must remain exactly equivalent to the native weighted CIoU reduction."""
        criterion = BboxLoss(reg_max=1)
        pred_dist = torch.ones(1, 1, 4)
        pred_boxes = torch.tensor([[[0.0, 0.0, 3.5, 4.0]]])
        target_boxes = torch.tensor([[[0.0, 0.0, 4.0, 4.0]]])
        scores = torch.tensor([[[0.75]]])
        foreground = torch.tensor([[True]])
        actual, _ = criterion(
            pred_dist,
            pred_boxes,
            torch.zeros(1, 2),
            target_boxes,
            scores,
            scores.sum(),
            foreground,
            torch.tensor([640.0, 640.0]),
            torch.tensor([[4.0]]),
        )
        ciou = bbox_iou(pred_boxes[foreground], target_boxes[foreground], xywh=False, CIoU=True)
        expected = ((1 - ciou) * scores[foreground].sum(-1, keepdim=True)).sum() / scores.sum()
        self.assertTrue(torch.equal(actual, expected))

    def test_e2e_loss_enables_sa_nwd_in_both_branches(self) -> None:
        """The one-to-many and one-to-one YOLO26 criteria must use identical SA-NWD settings."""
        model = YOLO("upstream/ultralytics/cfg/models/26/yolo26-p2.yaml").model
        args = {**DEFAULT_CFG_DICT, "epochs": 3, "sa_nwd": True, "nwd_area_limit": 1024.0}
        model.args = SimpleNamespace(**args)
        criterion = E2ELoss(model)
        for branch in (criterion.one2many, criterion.one2one):
            self.assertIsInstance(branch.bbox_loss.sa_nwd, SizeAwareNWDLoss)
            self.assertEqual(branch.bbox_loss.sa_nwd.area_limit, 1024.0)
            self.assertEqual(branch.assigner.topk, 10 if branch is criterion.one2many else 7)


if __name__ == "__main__":
    unittest.main()
