"""Unit and integration tests for shallow P2 channel attention."""

from __future__ import annotations

import io
import unittest
from pathlib import Path

import torch

from sdf.modules import DWF, IdentityChannelAttentionConcat, SDD
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
MODEL_YAML = ROOT / "configs/models/yolo26n-p2-identity-channel-attention.yaml"


class IdentityChannelAttentionConcatTest(unittest.TestCase):
    """Verify near-identity attention and its isolated P2 integration."""

    def test_identity_shape_gradient_and_serialization(self) -> None:
        """The initial gate must preserve shape/value, train, and round-trip exactly."""
        module = IdentityChannelAttentionConcat([32, 64])
        high = torch.randn(2, 32, 20, 24, requires_grad=True)
        shallow = torch.randn(2, 64, 20, 24, requires_grad=True)
        output = module([high, shallow])
        self.assertEqual(output.shape, (2, 96, 20, 24))
        self.assertTrue(torch.equal(output, torch.cat((high, shallow), 1)))

        output.square().mean().backward()
        self.assertTrue(torch.isfinite(high.grad).all())
        self.assertTrue(torch.isfinite(shallow.grad).all())
        gradients = (parameter.grad for parameter in module.parameters())
        self.assertTrue(all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients))

        buffer = io.BytesIO()
        torch.save(module.state_dict(), buffer)
        buffer.seek(0)
        restored = IdentityChannelAttentionConcat([32, 64])
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        with torch.no_grad():
            inputs = [high.detach(), shallow.detach()]
            self.assertTrue(torch.equal(module(inputs), restored(inputs)))

    def test_yolo_yaml_isolates_attention_to_official_p2_fusion(self) -> None:
        """The candidate must add one attention gate while retaining four detection strides."""
        model = YOLO(MODEL_YAML).model
        self.assertEqual(sum(isinstance(layer, IdentityChannelAttentionConcat) for layer in model.modules()), 1)
        self.assertEqual(sum(isinstance(layer, (DWF, SDD)) for layer in model.modules()), 0)
        self.assertEqual(model.stride.tolist(), [4.0, 8.0, 16.0, 32.0])
        with torch.no_grad():
            self.assertIsNotNone(model(torch.randn(1, 3, 640, 640)))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for AMP validation")
    def test_cuda_amp_backward(self) -> None:
        """Attention inputs and parameters must receive finite gradients under FP16 autocast."""
        module = IdentityChannelAttentionConcat([32, 64]).cuda()
        high = torch.randn(2, 32, 40, 40, device="cuda", requires_grad=True)
        shallow = torch.randn(2, 64, 40, 40, device="cuda", requires_grad=True)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = module([high, shallow]).square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(high.grad).all())
        self.assertTrue(torch.isfinite(shallow.grad).all())
        gradients = (parameter.grad for parameter in module.parameters())
        self.assertTrue(all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients))


if __name__ == "__main__":
    unittest.main()
