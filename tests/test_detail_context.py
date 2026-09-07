"""Unit and integration tests for residual P1 detail-context injection."""

from __future__ import annotations

import io
import unittest
from pathlib import Path

import torch

from sdf.modules import ResidualDetailContextConcat
from ultralytics.nn.tasks import DetectionModel


ROOT = Path(__file__).resolve().parents[1]
MODEL_YAML = ROOT / "configs/models/yolo26n-p2-rdci.yaml"
OFFICIAL_YAML = ROOT / "upstream/ultralytics/cfg/models/26/yolo26-p2.yaml"


class ResidualDetailContextConcatTest(unittest.TestCase):
    """Verify identity initialization, optimization, serialization, and YOLO integration."""

    @staticmethod
    def inputs(device: str = "cpu") -> list[torch.Tensor]:
        """Create top-down, P2, and P1 inputs at their expected relative resolutions."""
        return [
            torch.randn(2, 32, 20, 24, device=device, requires_grad=True),
            torch.randn(2, 64, 20, 24, device=device, requires_grad=True),
            torch.randn(2, 16, 40, 48, device=device, requires_grad=True),
        ]

    def test_identity_gradient_learning_and_serialization(self) -> None:
        """The module must start as concat, then expose finite gradients and round-trip state."""
        module = ResidualDetailContextConcat([32, 64, 16])
        inputs = self.inputs()
        expected = torch.cat(inputs[:2], dim=1)
        output = module(inputs)
        self.assertTrue(torch.equal(output, expected))

        optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
        output.square().mean().backward()
        self.assertTrue(all(tensor.grad is not None and torch.isfinite(tensor.grad).all() for tensor in inputs))
        self.assertTrue(
            all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in module.parameters())
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        learned_inputs = self.inputs()
        module(learned_inputs).square().mean().backward()
        self.assertGreater(float(module.align[1].weight.grad.abs().sum()), 0.0)

        buffer = io.BytesIO()
        torch.save(module.state_dict(), buffer)
        buffer.seek(0)
        restored = ResidualDetailContextConcat([32, 64, 16])
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        module.eval()
        restored.eval()
        with torch.no_grad():
            detached = [tensor.detach() for tensor in learned_inputs]
            self.assertTrue(torch.equal(module(detached), restored(detached)))

    def test_spatial_mismatch_is_rejected(self) -> None:
        """Invalid P1-to-P2 geometry must fail instead of broadcasting or resizing silently."""
        module = ResidualDetailContextConcat([32, 64, 16])
        inputs = self.inputs()
        inputs[2] = torch.randn(2, 16, 39, 48)
        with self.assertRaisesRegex(ValueError, "spatial mismatch"):
            module(inputs)

    def test_yolo_yaml_preserves_official_weights_and_strides(self) -> None:
        """RDCI must add only its own parameters while preserving every official P2 state key."""
        torch.manual_seed(0)
        official = DetectionModel(OFFICIAL_YAML, ch=3, nc=10, verbose=False)
        torch.manual_seed(0)
        candidate = DetectionModel(MODEL_YAML, ch=3, nc=10, verbose=False)
        self.assertEqual(sum(isinstance(layer, ResidualDetailContextConcat) for layer in candidate.modules()), 1)
        self.assertEqual(candidate.stride.tolist(), [4.0, 8.0, 16.0, 32.0])

        official_state = official.state_dict()
        candidate_state = candidate.state_dict()
        incompatible = candidate.load_state_dict(official_state, strict=False)
        self.assertFalse(incompatible.unexpected_keys)
        self.assertTrue(set(official_state).issubset(candidate_state))
        self.assertTrue(all(key.startswith("model.18.") for key in set(candidate_state) - set(official_state)))
        self.assertTrue(all(key.startswith("model.18.") for key in incompatible.missing_keys))
        official.eval()
        candidate.eval()
        with torch.no_grad():
            image = torch.randn(1, 3, 64, 64)
            self.assertTrue(torch.equal(official(image)[0], candidate(image)[0]))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for AMP validation")
    def test_cuda_amp_backward(self) -> None:
        """RDCI inputs and parameters must receive finite gradients under CUDA autocast."""
        module = ResidualDetailContextConcat([32, 64, 16]).cuda()
        inputs = self.inputs("cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            loss = module(inputs).square().mean()
        loss.backward()
        self.assertTrue(all(tensor.grad is not None and torch.isfinite(tensor.grad).all() for tensor in inputs))
        self.assertTrue(
            all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in module.parameters())
        )


if __name__ == "__main__":
    unittest.main()
