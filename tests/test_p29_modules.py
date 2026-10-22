"""Regression tests for the P29 CoordAtt, RFBlock, and EMAAttention modules."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import torch

from sdf.modules import CoordAtt, EMAAttention, RFBlock


def _load_reference_ema():
    """Load the verbatim upstream EMA implementation stored as a test fixture."""
    fixture = Path(__file__).parent / "fixtures" / "ema_reference.py"
    spec = importlib.util.spec_from_file_location("ema_reference", fixture)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.EMA


class TestCoordAtt(unittest.TestCase):
    def test_shape_preserved(self):
        module = CoordAtt(64)
        x = torch.randn(2, 64, 160, 160)
        self.assertEqual(module(x).shape, x.shape)

    def test_output_bounded_by_input(self):
        module = CoordAtt(64).eval()
        x = torch.randn(2, 64, 40, 40) * 5
        with torch.no_grad():
            out = module(x)
        self.assertTrue(torch.all(out.abs() <= x.abs() + 1e-6))

    def test_param_count_documented(self):
        module = CoordAtt(64)
        self.assertEqual(sum(p.numel() for p in module.parameters()), 1552)

    def test_gradient_flows(self):
        x = torch.randn(1, 64, 32, 32, requires_grad=True)
        CoordAtt(64)(x).sum().backward()
        self.assertGreater(x.grad.abs().sum().item(), 0.0)


class TestRFBlock(unittest.TestCase):
    def test_shape_preserved(self):
        module = RFBlock(64)
        x = torch.randn(2, 64, 160, 160)
        self.assertEqual(module(x).shape, x.shape)

    def test_identity_at_zero_branch_weights(self):
        module = RFBlock(64).eval()
        for branch in module.branches:
            torch.nn.init.zeros_(branch[-2].weight)  # zero the 3x3 branch conv
            torch.nn.init.zeros_(branch[-1].weight)  # and its BN gamma
            torch.nn.init.zeros_(branch[-1].bias)
        torch.nn.init.zeros_(module.fuse[0].weight)
        torch.nn.init.zeros_(module.fuse[1].weight)
        torch.nn.init.zeros_(module.fuse[1].bias)
        x = torch.randn(2, 64, 32, 32)
        with torch.no_grad():
            out = module(x)
        self.assertTrue(torch.allclose(out, x, atol=1e-6))

    def test_param_count_documented(self):
        module = RFBlock(64)
        self.assertEqual(sum(p.numel() for p in module.parameters()), 13376)

    def test_gradient_flows(self):
        x = torch.randn(1, 64, 32, 32, requires_grad=True)
        RFBlock(64)(x).sum().backward()
        self.assertGreater(x.grad.abs().sum().item(), 0.0)


class TestEMAAttention(unittest.TestCase):
    def test_shape_preserved(self):
        module = EMAAttention(64)
        x = torch.randn(2, 64, 160, 160)
        self.assertEqual(module(x).shape, x.shape)

    def test_output_bounded_by_input(self):
        module = EMAAttention(64).eval()
        x = torch.randn(2, 64, 40, 40) * 5
        with torch.no_grad():
            out = module(x)
        self.assertTrue(torch.all(out.abs() <= x.abs() + 1e-6))

    def test_param_count_documented(self):
        module = EMAAttention(64)
        self.assertEqual(sum(p.numel() for p in module.parameters()), 48)

    def test_gradient_flows(self):
        x = torch.randn(1, 64, 32, 32, requires_grad=True)
        EMAAttention(64)(x).sum().backward()
        self.assertGreater(x.grad.abs().sum().item(), 0.0)

    def test_matches_reference_implementation(self):
        reference_cls = _load_reference_ema()
        torch.manual_seed(29)
        ours = EMAAttention(64).eval()
        reference = reference_cls(64).eval()
        reference.load_state_dict(ours.state_dict())
        x = torch.randn(2, 64, 48, 80)
        with torch.no_grad():
            out_ours = ours(x)
            out_ref = reference(x)
        self.assertTrue(torch.allclose(out_ours, out_ref, atol=1e-6))
        self.assertTrue(torch.equal(out_ours, out_ref))


class TestModelIntegration(unittest.TestCase):
    def test_model_deltas(self):
        from ultralytics import YOLO

        base = sum(p.numel() for p in YOLO("configs/models/yolo26n-p2-acr-acfr.yaml", task="detect").model.parameters())
        ca = sum(p.numel() for p in YOLO("configs/models/yolo26n-p2-acr-acfr-ca.yaml", task="detect").model.parameters())
        rfb = sum(p.numel() for p in YOLO("configs/models/yolo26n-p2-acr-acfr-rfb.yaml", task="detect").model.parameters())
        ema = sum(p.numel() for p in YOLO("configs/models/yolo26n-p2-acr-acfr-ema.yaml", task="detect").model.parameters())
        self.assertEqual(ca - base, 1552)
        self.assertEqual(rfb - base, 13376)
        self.assertEqual(ema - base, 48)
        self.assertEqual(type(YOLO("configs/models/yolo26n-p2-acr-acfr-ca.yaml", task="detect").model.model[20]).__name__, "CoordAtt")
        self.assertEqual(type(YOLO("configs/models/yolo26n-p2-acr-acfr-ema.yaml", task="detect").model.model[20]).__name__, "EMAAttention")


if __name__ == "__main__":
    unittest.main()
