"""Regression tests for P2-teacher to P3-student small-object distillation."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import torch

from ultralytics.cfg import get_cfg
from ultralytics.nn.distill_model import DistillationModel, build_small_object_mask
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import strip_optimizer


ROOT = Path(__file__).resolve().parents[1]
STUDENT_YAML = ROOT / "upstream/ultralytics/cfg/models/26/yolo26.yaml"
TEACHER_YAML = ROOT / "upstream/ultralytics/cfg/models/26/yolo26-p2.yaml"


def build_model(mode: str = "p2p3_consensus", device: str = "cpu") -> DistillationModel:
    """Build a compact-input native student and P2 teacher pair."""
    student = DetectionModel(STUDENT_YAML, ch=3, nc=10, verbose=False).to(device)
    teacher = DetectionModel(TEACHER_YAML, ch=3, nc=10, verbose=False).to(device)
    student.args = get_cfg(overrides={"imgsz": 64, "dis": 6.0, "distill_mode": mode})
    return DistillationModel(teacher, student)


def synthetic_batch(device: str = "cpu") -> dict[str, torch.Tensor]:
    """Return a two-image detection batch with one small and one larger object."""
    return {
        "img": torch.rand(2, 3, 64, 64, device=device),
        "batch_idx": torch.tensor([0.0, 1.0], device=device),
        "cls": torch.tensor([[0.0], [1.0]], device=device),
        "bboxes": torch.tensor([[0.5, 0.5, 0.08, 0.08], [0.3, 0.4, 0.2, 0.15]], device=device),
    }


class P2P3DistillationTest(unittest.TestCase):
    """Verify mapping, weighting, gradients, serialization, and mixed precision."""

    def test_small_object_mask_is_finite_monotonic_and_empty_safe(self) -> None:
        """Smaller boxes must receive greater weight and empty images must remain finite."""
        batch = {
            "img": torch.zeros(2, 3, 640, 640),
            "batch_idx": torch.tensor([0.0, 1.0]),
            "bboxes": torch.tensor([[0.5, 0.5, 0.02, 0.02], [0.5, 0.5, 0.2, 0.2]]),
        }
        mask = build_small_object_mask(batch, (80, 80), torch.float32)
        self.assertEqual(mask.shape, (2, 1, 80, 80))
        self.assertTrue(torch.isfinite(mask).all())
        self.assertGreater(float(mask[0].max()), float(mask[1].max()))
        self.assertTrue(torch.equal(mask, build_small_object_mask(batch, (80, 80), torch.float32)))

        reference_box = torch.tensor([[0.5, 0.5, 0.08, 0.08]])
        mask_640 = build_small_object_mask(
            {"img": torch.zeros(1, 3, 640, 640), "batch_idx": torch.tensor([0.0]), "bboxes": reference_box},
            (80, 80),
            torch.float32,
        )
        mask_960 = build_small_object_mask(
            {"img": torch.zeros(1, 3, 960, 960), "batch_idx": torch.tensor([0.0]), "bboxes": reference_box},
            (120, 120),
            torch.float32,
        )
        self.assertEqual(float(mask_640.max()), float(mask_960.max()))

        empty = {"img": torch.zeros(2, 3, 640, 640), "batch_idx": torch.empty(0), "bboxes": torch.empty(0, 4)}
        empty_mask = build_small_object_mask(empty, (80, 80), torch.float16)
        self.assertEqual(empty_mask.dtype, torch.float16)
        self.assertEqual(int(torch.count_nonzero(empty_mask)), 0)

    def test_mapping_teacher_freeze_backward_and_deepcopy(self) -> None:
        """The bridge must align all levels and train only student/projector parameters."""
        model = build_model().train()
        self.assertEqual(model.student_feats_idx, [16, 19, 22, 23])
        self.assertEqual(model.teacher_feats_idx, [19, 22, 25, 28, 29])
        self.assertTrue(all(not parameter.requires_grad for parameter in model.teacher_model.parameters()))
        self.assertFalse(model.teacher_model.training)

        teacher_feats = [model.decouple_outputs(model._teacher_feats[index]) for index in model.teacher_feats_idx[:-1]]
        student_feats = [model.decouple_outputs(model._student_feats[index]) for index in model.student_feats_idx[:-1]]
        targets = model._teacher_targets(teacher_feats)
        self.assertEqual([feature.shape[-2:] for feature in targets], [feature.shape[-2:] for feature in student_feats])
        self.assertEqual([layer[-1].out_channels for layer in model.projector], [96, 128, 256])

        loss, items = model.loss(synthetic_batch())
        self.assertTrue(torch.isfinite(loss).all())
        self.assertTrue(torch.isfinite(items["dis_loss"]).all())
        loss.sum().backward()
        self.assertTrue(all(parameter.grad is not None for parameter in model.projector.parameters()))
        self.assertTrue(any(parameter.grad is not None for parameter in model.student_model.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in model.teacher_model.parameters()))

        restored = copy.deepcopy(model)
        self.assertEqual(restored.distill_mode, "p2p3_consensus")
        self.assertEqual(restored.student_feats_idx, model.student_feats_idx)
        self.assertEqual(restored.teacher_feats_idx, model.teacher_feats_idx)
        self.assertEqual(len(restored._student_hooks), len(restored.student_feats_idx))
        self.assertEqual(len(restored._teacher_hooks), len(restored.teacher_feats_idx))

        ema = copy.deepcopy(model).eval()
        ema.teacher_model = None
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "best.pt"
            torch.save({"model": None, "ema": ema, "train_args": {}}, checkpoint_path)
            checkpoint = strip_optimizer(checkpoint_path)
        self.assertIsInstance(checkpoint["model"], DetectionModel)
        self.assertNotIsInstance(checkpoint["model"], DistillationModel)
        self.assertIsNone(checkpoint["ema"])

    def test_common_mapping_skips_teacher_p2(self) -> None:
        """The D1 control must compare only common P3/P4/P5 levels."""
        model = build_model("p2p3_common")
        teacher_feats = [model.decouple_outputs(model._teacher_feats[index]) for index in model.teacher_feats_idx[:-1]]
        targets = model._teacher_targets(teacher_feats)
        self.assertEqual([target.shape for target in targets], [feature.shape for feature in teacher_feats[1:]])
        self.assertEqual([layer[-1].out_channels for layer in model.projector], [64, 128, 256])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for AMP validation")
    def test_cuda_amp_backward(self) -> None:
        """Consensus distillation must produce finite gradients under CUDA autocast."""
        model = build_model(device="cuda").train()
        with torch.autocast("cuda", dtype=torch.float16):
            loss, items = model.loss(synthetic_batch("cuda"))
        loss.sum().backward()
        self.assertTrue(torch.isfinite(loss).all())
        self.assertTrue(torch.isfinite(items["dis_loss"]).all())
        self.assertTrue(
            all(
                parameter.grad is not None and torch.isfinite(parameter.grad).all()
                for parameter in model.projector.parameters()
            )
        )


if __name__ == "__main__":
    unittest.main()
