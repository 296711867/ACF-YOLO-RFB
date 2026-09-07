"""Unit and integration tests for SDD."""

from __future__ import annotations

import io
import unittest
from pathlib import Path

import torch

from sdf.modules import SDD
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
MODEL_YAML = ROOT / "configs/models/yolo26n-p2-sdd.yaml"


class SDDTest(unittest.TestCase):
    """Verify SDD invariants and YOLO integration."""

    def test_space_to_depth_is_lossless(self) -> None:
        """Pixel unshuffle must retain every input value."""
        x = torch.arange(2 * 3 * 8 * 10, dtype=torch.float32).reshape(2, 3, 8, 10)
        rearranged = SDD(3, 8).space_to_depth(x)
        restored = torch.nn.functional.pixel_shuffle(rearranged, 2)
        self.assertTrue(torch.equal(restored, x))

    def test_sdd_shape_gradient_and_serialization(self) -> None:
        """SDD must downsample exactly, backpropagate finite gradients, and round-trip its state."""
        module = SDD(16, 32)
        x = torch.randn(2, 16, 64, 96, requires_grad=True)
        y = module(x)
        self.assertEqual(y.shape, (2, 32, 32, 48))
        y.square().mean().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(
            all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in module.parameters())
        )

        buffer = io.BytesIO()
        torch.save(module.state_dict(), buffer)
        buffer.seek(0)
        restored = SDD(16, 32)
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        module.eval()
        restored.eval()
        with torch.no_grad():
            self.assertTrue(torch.equal(module(x.detach()), restored(x.detach())))

    def test_yolo_yaml_builds_expected_p2_model(self) -> None:
        """The custom YAML must retain the official four detection strides and use SDD only once."""
        model = YOLO(MODEL_YAML).model
        self.assertEqual(sum(isinstance(layer, SDD) for layer in model.modules()), 1)
        self.assertEqual(model.stride.tolist(), [4.0, 8.0, 16.0, 32.0])
        with torch.no_grad():
            output = model(torch.randn(1, 3, 640, 640))
        self.assertIsNotNone(output)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for AMP validation")
    def test_sdd_cuda_amp_backward(self) -> None:
        """SDD gradients must remain finite under CUDA autocast."""
        module = SDD(16, 32).cuda()
        x = torch.randn(2, 16, 64, 64, device="cuda", requires_grad=True)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = module(x).square().mean()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(
            all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in module.parameters())
        )


if __name__ == "__main__":
    unittest.main()
