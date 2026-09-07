"""Unit and integration tests for fixed Haar wavelet downsampling."""

from __future__ import annotations

import unittest
from pathlib import Path

import torch

from sdf.modules import HaarDownsample, haar_dwt, haar_idwt
from ultralytics.nn.tasks import DetectionModel


ROOT = Path(__file__).resolve().parents[1]
MODEL_YAML = ROOT / "configs/models/yolo26n-p2-hwd.yaml"
OFFICIAL_YAML = ROOT / "upstream/ultralytics/cfg/models/26/yolo26-p2.yaml"


class HaarDwtTest(unittest.TestCase):
    """Verify DWT orthonormality, exact inversion, and odd-size handling."""

    def test_shape_energy_and_reconstruction(self) -> None:
        """The DWT must be orthonormal and exactly invertible on even sizes."""
        torch.manual_seed(0)
        x = torch.randn(2, 16, 20, 24)
        subbands = haar_dwt(x)
        self.assertEqual(subbands.shape, (2, 64, 10, 12))
        self.assertTrue(torch.allclose(subbands.pow(2).sum(), x.pow(2).sum(), rtol=1e-4, atol=1e-4))
        self.assertTrue(torch.allclose(haar_idwt(subbands), x, rtol=1e-5, atol=1e-6))

    def test_known_values(self) -> None:
        """A single 2x2 block must produce the analytic Haar subbands."""
        x = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        subbands = haar_dwt(x)
        self.assertEqual(subbands.shape, (1, 4, 1, 1))
        self.assertTrue(torch.allclose(subbands.flatten(), torch.tensor([5.0, -1.0, -2.0, 0.0])))
        self.assertTrue(torch.equal(haar_idwt(subbands), x))

    def test_odd_spatial_size(self) -> None:
        """Odd inputs are edge-padded once and downsampled with ceil semantics."""
        torch.manual_seed(1)
        x = torch.randn(2, 3, 7, 5)
        subbands = haar_dwt(x)
        self.assertEqual(subbands.shape, (2, 12, 4, 3))


class HaarDownsampleTest(unittest.TestCase):
    """Verify module shape contract, gradients, and serialization."""

    def test_forward_backward_and_state_roundtrip(self) -> None:
        """The module must halve resolution, keep gradients finite, and round-trip state."""
        torch.manual_seed(0)
        module = HaarDownsample(16, 32, kernel_size=3)
        x = torch.randn(2, 16, 40, 44, requires_grad=True)
        output = module(x)
        self.assertEqual(output.shape, (2, 32, 20, 22))
        output.square().mean().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(
            all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
        )
        clone = HaarDownsample(16, 32, kernel_size=3)
        clone.load_state_dict(module.state_dict())
        with torch.no_grad():
            self.assertTrue(torch.equal(clone(x), output))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA device required")
    def test_cuda_amp_forward_backward(self) -> None:
        """FP16 autocast forward and backward must stay finite on CUDA."""
        torch.manual_seed(0)
        module = HaarDownsample(16, 32, kernel_size=3).cuda()
        x = torch.randn(2, 16, 80, 88, device="cuda", requires_grad=True)
        with torch.autocast("cuda", dtype=torch.float16):
            output = module(x)
        self.assertEqual(output.shape, (2, 32, 40, 44))
        output.float().square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())


class HwdModelIntegrationTest(unittest.TestCase):
    """Verify the YAML builds the intended network relative to official YOLO26n-P2."""

    def test_structure_stride_and_weight_compatibility(self) -> None:
        """Only layers 0/1 differ; strides and all deeper weights must match the official model."""
        official = DetectionModel(OFFICIAL_YAML, ch=3, nc=10, verbose=False)
        candidate = DetectionModel(MODEL_YAML, ch=3, nc=10, verbose=False)
        self.assertEqual(sum(isinstance(layer, HaarDownsample) for layer in candidate.modules()), 2)
        self.assertEqual(candidate.stride.tolist(), [4.0, 8.0, 16.0, 32.0])

        official_params = sum(p.numel() for p in official.parameters())
        candidate_params = sum(p.numel() for p in candidate.parameters())
        delta = candidate_params - official_params
        self.assertGreater(delta, 0)
        self.assertLess(delta, 30_000)

        official_state = official.state_dict()
        incompatible = candidate.load_state_dict(official_state, strict=False)
        replaced = {key for key in official_state if key.startswith(("model.0.", "model.1."))}
        self.assertEqual(set(incompatible.unexpected_keys), replaced)
        self.assertTrue(all(key.startswith(("model.0.fusion.", "model.1.fusion.")) for key in incompatible.missing_keys))
        self.assertTrue(set(official_state) - replaced <= set(candidate.state_dict()))

        torch.manual_seed(0)
        candidate.eval()
        with torch.no_grad():
            outputs = candidate(torch.randn(1, 3, 640, 640))
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        predictions = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
        if predictions.ndim == 3:
            self.assertEqual(predictions.shape[0], 1)
            predictions = predictions[0]
        self.assertEqual(predictions.ndim, 2)
        self.assertEqual(predictions.shape[1], 6)
        self.assertTrue(torch.isfinite(predictions).all())


if __name__ == "__main__":
    unittest.main()
