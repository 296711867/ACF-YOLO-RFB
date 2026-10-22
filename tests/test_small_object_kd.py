"""Unit tests for the P22 small-object-aware KD modes (K1/K2/K3)."""

from __future__ import annotations

import unittest
from pathlib import Path

import torch
from torch import nn
from ultralytics.cfg import get_cfg
from ultralytics.nn.distill_model import (
    SMALL_KD_LAMBDA_BOX,
    SMALL_KD_LAMBDA_CLS,
    SMALL_KD_LAMBDA_FEAT,
    SMALL_KD_MGD_ALPHA,
    SMALL_KD_MGD_MASK_RATIO,
    SMALL_KD_TEMPERATURE,
    DistillationModel,
    build_small_object_mask,
)
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import YAML


ROOT = Path(__file__).resolve().parents[1]
STUDENT_YAML = ROOT / "configs/models/yolo26n-p2-acr-acfr.yaml"
TEACHER_YAML = ROOT / "configs/models/yolo26s-p2-teacher.yaml"


def tiny_args(**overrides):
    base = {
        "model": str(STUDENT_YAML),
        "data": "configs/data/bdd100k.yaml",
        "imgsz": 64,
        "batch": 2,
        "device": "cpu",
        "distill_mode": "small_kd_joint",
        "dis": 1.0,
    }
    base.update(overrides)
    args = get_cfg(overrides={k: v for k, v in base.items() if k != "stage_stop_epoch"})
    # get_cfg only keeps known keys; re-attach the distillation keys it drops.
    args.distill_mode = base["distill_mode"]
    args.dis = base["dis"]
    args.imgsz = base["imgsz"]
    return args


def synthetic_batch(batch_size: int = 2, imgsz: int = 64) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    count = batch_size * 4
    centers = 0.2 + 0.6 * torch.rand(count, 2, generator=generator)
    sizes = 0.02 + 0.04 * torch.rand(count, 2, generator=generator)  # small objects
    return {
        "img": torch.rand(batch_size, 3, imgsz, imgsz, generator=generator),
        "batch_idx": torch.arange(batch_size).repeat_interleave(4).float(),
        "cls": (torch.arange(count) % 10).float().view(-1, 1),
        "bboxes": torch.cat((centers, sizes), dim=1),
    }


class SmallObjectKdTest(unittest.TestCase):
    """K1/K2/K3 modes: constants, mask, loss computation, gradients, teacher freeze."""

    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(0)
        cls.student = DetectionModel(str(STUDENT_YAML), ch=3, nc=10, verbose=False)
        cls.student.args = tiny_args()
        torch.manual_seed(1)
        cls.teacher = DetectionModel(str(TEACHER_YAML), ch=3, nc=10, verbose=False)
        cls.teacher.eval()

    def _build(self, mode: str) -> DistillationModel:
        self.student.args = tiny_args(distill_mode=mode)
        torch.manual_seed(0)
        return DistillationModel(teacher_model=self.teacher, student_model=self.student)

    def test_frozen_protocol_constants(self) -> None:
        """Lambdas and temperature match the frozen P22 protocol."""
        self.assertEqual((SMALL_KD_LAMBDA_CLS, SMALL_KD_LAMBDA_BOX, SMALL_KD_LAMBDA_FEAT), (1.0, 0.5, 0.1))
        self.assertEqual(SMALL_KD_TEMPERATURE, 2.0)

    def test_small_object_mask_shape_and_weighting(self) -> None:
        """Mask has the requested grid shape and weights small boxes above large ones."""
        batch = {
            "img": torch.rand(2, 3, 64, 64),
            "batch_idx": torch.tensor([0.0, 1.0]),
            "cls": torch.tensor([[0.0], [0.0]]),
            "bboxes": torch.tensor([[0.5, 0.5, 0.02, 0.02], [0.5, 0.5, 0.50, 0.50]]),
        }
        mask = build_small_object_mask(batch, (8, 8), torch.float32)
        self.assertEqual(mask.shape, (2, 1, 8, 8))
        self.assertGreater(float(mask[0].max()), 0.8)  # ~52 px area -> exp(-52/1024)=0.95 scale
        self.assertLess(float(mask[1].max()), 1e-6)  # 102400 px area -> ~0

    def test_all_three_modes_compute_finite_loss_with_grads(self) -> None:
        """K1/K2/K3 produce finite loss, finite student grads, frozen teacher."""
        batch = synthetic_batch()
        for mode in ("small_kd_out", "small_kd_feat", "small_kd_joint"):
            model = self._build(mode)
            model.train()
            loss, items = model.loss(batch)
            total = loss.sum()
            self.assertTrue(torch.isfinite(total).all(), mode)
            self.assertIn("dis_loss", items)
            self.assertGreater(float(items["dis_loss"]), 0.0, mode)
            total.backward()
            student_grads = [p.grad for p in model.student_model.parameters() if p.grad is not None]
            self.assertTrue(student_grads, mode)
            self.assertTrue(all(torch.isfinite(g).all() for g in student_grads), mode)
            self.assertTrue(all(p.grad is None for p in model.teacher_model.parameters()), mode)
            model.zero_grad()

    def test_projector_covers_two_to_one_channels(self) -> None:
        """The s-scale teacher doubles channels; projectors must map student dims up."""
        model = self._build("small_kd_joint")
        student_dims = [proj[0].in_channels for proj in model.projector]
        teacher_dims = [proj[-1].out_channels for proj in model.projector]
        self.assertEqual(student_dims, [64, 64, 128, 192])
        # Official s-P2 widths: P2=64 (matches the ACR-widened student), P3/P4/P5 double.
        self.assertEqual(teacher_dims, [64, 128, 256, 512])

    def test_eval_loss_has_no_distillation_term(self) -> None:
        """Validation path returns the regular loss with zero distillation."""
        batch = synthetic_batch()
        model = self._build("small_kd_joint")
        model.eval()
        loss, items = model.loss(batch)
        self.assertEqual(float(items["dis_loss"]), 0.0)

    def test_unknown_mode_rejected(self) -> None:
        """An unrecognized distill_mode raises before any training starts."""
        with self.assertRaises(ValueError):
            self._build("small_kd_bogus")

    def test_mgd_frozen_constants(self) -> None:
        """P23 MGD constants match the frozen protocol: mask 0.5, alpha 2e-5."""
        self.assertEqual((SMALL_KD_MGD_MASK_RATIO, SMALL_KD_MGD_ALPHA), (0.5, 2e-5))

    def test_mgd_mode_computes_finite_loss_with_grads(self) -> None:
        """M1 (small_kd_mgd) produces finite loss, student+head grads, frozen teacher."""
        batch = synthetic_batch()
        model = self._build("small_kd_mgd")
        model.train()
        loss, items = model.loss(batch)
        total = loss.sum()
        self.assertTrue(torch.isfinite(total).all())
        self.assertGreater(float(items["dis_loss"]), 0.0)
        total.backward()
        student_grads = [p.grad for p in model.student_model.parameters() if p.grad is not None]
        self.assertTrue(student_grads)
        self.assertTrue(all(torch.isfinite(g).all() for g in student_grads))
        self.assertTrue(all(p.grad is None for p in model.teacher_model.parameters()))
        head_grads = [p.grad for p in model.mgd_generation.parameters() if p.grad is not None]
        self.assertTrue(head_grads)  # the generation head itself trains
        model.zero_grad()

    def test_mgd_generation_heads_cover_p2_p3(self) -> None:
        """Two heads only (P2/P3); 1x1 align maps student dims to teacher dims."""
        model = self._build("small_kd_mgd")
        self.assertEqual(len(model.mgd_generation), 2)
        self.assertEqual([head[0].in_channels for head in model.mgd_generation], [64, 64])
        self.assertEqual([head[-1].out_channels for head in model.mgd_generation], [64, 128])

    def test_mgd_masked_sq_sum_math(self) -> None:
        """Zero on perfect reconstruction, positive otherwise, only on dropped pixels."""
        x = torch.rand(2, 4, 8, 8)
        dropped = torch.zeros(2, 1, 8, 8)
        dropped[:, :, :4] = 1.0  # top half dropped
        self.assertAlmostEqual(float(DistillationModel._mgd_masked_sq_sum(x, x, dropped)), 0.0, places=6)
        self.assertGreater(float(DistillationModel._mgd_masked_sq_sum(torch.zeros_like(x), torch.ones_like(x), dropped)), 0.0)
        # errors outside dropped pixels must not contribute
        a = torch.zeros(2, 4, 8, 8)
        b = torch.zeros(2, 4, 8, 8)
        b[:, :, 4:] = 5.0  # error only in the KEPT region
        self.assertAlmostEqual(float(DistillationModel._mgd_masked_sq_sum(a, b, dropped)), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
