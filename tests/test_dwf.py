"""Unit and integration tests for DWF."""

from __future__ import annotations

import io
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from sdf.modules import DWF, DWFNearestGate, DWFNearestWeight, DySample, SDD
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
MODEL_YAML = ROOT / "configs/models/yolo26n-p2-dwf.yaml"
NEAREST_GATE_MODEL_YAML = ROOT / "configs/models/yolo26n-p2-dwf-nearest-gate.yaml"
NEAREST_WEIGHT_MODEL_YAML = ROOT / "configs/models/yolo26n-p2-dwf-nearest-weight.yaml"


class DWFTest(unittest.TestCase):
    """Verify dynamic sampling, fusion invariants, and YOLO integration."""

    def test_zero_offset_sampling_matches_bilinear(self) -> None:
        """A newly initialized DySample must start from standard bilinear upsampling."""
        x = torch.randn(2, 8, 9, 13)
        actual = DySample(8)(x)
        expected = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-5, rtol=1e-5))

    def test_dwf_weights_shape_gradient_and_serialization(self) -> None:
        """DWF weights, output shape, gradients, and state round-trip must be valid."""
        module = DWF([64, 64], 64)
        weights = module.normalized_weights()
        self.assertTrue(torch.all(weights >= 0))
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=3)

        high = torch.randn(2, 64, 20, 24, requires_grad=True)
        low = torch.randn(2, 64, 40, 48, requires_grad=True)
        output = module([high, low])
        self.assertEqual(output.shape, low.shape)
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(high.grad).all())
        self.assertTrue(torch.isfinite(low.grad).all())
        self.assertTrue(
            all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in module.parameters())
        )

        buffer = io.BytesIO()
        torch.save(module.state_dict(), buffer)
        buffer.seek(0)
        restored = DWF([64, 64], 64)
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        module.eval()
        restored.eval()
        with torch.no_grad():
            self.assertTrue(torch.equal(module([high.detach(), low.detach()]), restored([high.detach(), low.detach()])))

    def test_yolo_yaml_builds_expected_p2_model(self) -> None:
        """The custom YAML must contain only DWF and retain four detection strides."""
        model = YOLO(MODEL_YAML).model
        self.assertEqual(sum(isinstance(layer, DWF) for layer in model.modules()), 1)
        self.assertEqual(sum(isinstance(layer, SDD) for layer in model.modules()), 0)
        self.assertEqual(model.stride.tolist(), [4.0, 8.0, 16.0, 32.0])
        with torch.no_grad():
            output = model(torch.randn(1, 3, 640, 640))
        self.assertIsNotNone(output)

    def test_nearest_gate_variant_excludes_dynamic_sampling(self) -> None:
        """Variant A must retain DWF fusion while removing every dynamic sampler."""
        module = DWFNearestGate([64, 64], 64)
        self.assertEqual(sum(isinstance(layer, DySample) for layer in module.modules()), 0)
        high = torch.randn(2, 64, 20, 24, requires_grad=True)
        low = torch.randn(2, 64, 40, 48, requires_grad=True)
        output = module([high, low])
        self.assertEqual(output.shape, low.shape)
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(high.grad).all())
        self.assertTrue(torch.isfinite(low.grad).all())

        model = YOLO(NEAREST_GATE_MODEL_YAML).model
        self.assertEqual(sum(isinstance(layer, DWFNearestGate) for layer in model.modules()), 1)
        self.assertEqual(sum(isinstance(layer, DySample) for layer in model.modules()), 0)
        self.assertEqual(sum(isinstance(layer, SDD) for layer in model.modules()), 0)
        self.assertEqual(model.stride.tolist(), [4.0, 8.0, 16.0, 32.0])

    def test_nearest_weight_variant_excludes_dynamic_sampling_and_gate(self) -> None:
        """Variant B must retain global weighting while removing dynamic sampling and local gating."""
        module = DWFNearestWeight([64, 64], 64)
        self.assertFalse(hasattr(module, "gate"))
        self.assertEqual(sum(isinstance(layer, DySample) for layer in module.modules()), 0)
        high = torch.randn(2, 64, 20, 24, requires_grad=True)
        low = torch.randn(2, 64, 40, 48, requires_grad=True)
        output = module([high, low])
        self.assertEqual(output.shape, low.shape)
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(high.grad).all())
        self.assertTrue(torch.isfinite(low.grad).all())
        self.assertIsNotNone(module.scale_weights.grad)

        model = YOLO(NEAREST_WEIGHT_MODEL_YAML).model
        self.assertEqual(sum(isinstance(layer, DWFNearestWeight) for layer in model.modules()), 1)
        self.assertEqual(sum(isinstance(layer, DySample) for layer in model.modules()), 0)
        self.assertEqual(sum(hasattr(layer, "gate") for layer in model.modules()), 0)
        self.assertEqual(sum(isinstance(layer, SDD) for layer in model.modules()), 0)
        self.assertEqual(model.stride.tolist(), [4.0, 8.0, 16.0, 32.0])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for AMP validation")
    def test_dwf_cuda_amp_backward(self) -> None:
        """DWF gradients must remain finite under CUDA autocast."""
        module = DWF([64, 64], 64).cuda()
        high = torch.randn(2, 64, 20, 20, device="cuda", requires_grad=True)
        low = torch.randn(2, 64, 40, 40, device="cuda", requires_grad=True)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = module([high, low]).square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(high.grad).all())
        self.assertTrue(torch.isfinite(low.grad).all())
        self.assertTrue(
            all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in module.parameters())
        )


if __name__ == "__main__":
    unittest.main()
