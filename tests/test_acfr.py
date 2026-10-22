"""Unit and integration tests for ACFR adaptive complementary fusion (P19A)."""

from __future__ import annotations

import unittest
from pathlib import Path

import torch

from sdf.modules import ACFRConcat
from ultralytics.nn.tasks import DetectionModel


ROOT = Path(__file__).resolve().parents[1]
ACR_YAML = ROOT / "configs/models/yolo26n-p2-acr.yaml"
CANDIDATE_YAML = ROOT / "configs/models/yolo26n-p2-acr-acfr.yaml"


class ACFRConcatModuleTest(unittest.TestCase):
    """Verify identity initialization, error exposure, and serialization."""

    def test_identity_shape_static_parameters_and_backward(self) -> None:
        """At init the module equals Concat bitwise; gradients exist and stay finite."""
        torch.manual_seed(0)
        module = ACFRConcat([64, 64], groups=4, refine_ratio=0.5, reduction=4)
        high = torch.randn(2, 64, 40, 40, requires_grad=True)
        low = torch.randn(2, 64, 40, 40, requires_grad=True)
        parameter_ids = tuple(id(parameter) for parameter in module.parameters())

        output = module([high, low])
        self.assertEqual(output.shape, (2, 128, 40, 40))
        self.assertTrue(torch.equal(output, torch.cat((high, low), dim=1)))
        self.assertEqual(parameter_ids, tuple(id(parameter) for parameter in module.parameters()))

        output.square().mean().backward()
        self.assertTrue(torch.isfinite(high.grad).all())
        self.assertTrue(torch.isfinite(low.grad).all())
        self.assertTrue(
            all(
                parameter.grad is not None and torch.isfinite(parameter.grad).all()
                for parameter in module.parameters()
            )
        )

    def test_constructor_and_forward_error_exposure(self) -> None:
        """Invalid arity, channel mismatch, group indivisibility, and spatial mismatch raise."""
        with self.assertRaises(ValueError):
            ACFRConcat([64, 32])
        with self.assertRaises(ValueError):
            ACFRConcat([64, 64], groups=5)
        with self.assertRaises(ValueError):
            ACFRConcat([0, 0])
        module = ACFRConcat([64, 64], groups=4, refine_ratio=0.5, reduction=4)
        with self.assertRaises(ValueError):
            module([torch.randn(1, 64, 8, 8)])
        with self.assertRaises(ValueError):
            module([torch.randn(1, 64, 8, 8), torch.randn(1, 32, 8, 8)])
        with self.assertRaises(ValueError):
            module([torch.randn(1, 64, 8, 8), torch.randn(1, 64, 9, 9)])

    def test_state_roundtrip_and_optimizer_coverage(self) -> None:
        """Saved state restores exact outputs; optimizer sees every trainable parameter."""
        torch.manual_seed(1)
        module = ACFRConcat([64, 64])
        inputs = [torch.randn(2, 64, 16, 16), torch.randn(2, 64, 16, 16)]
        with torch.no_grad():
            for parameter in module.gate.parameters():
                parameter.add_(0.01)
            for residual in (module.high_refine.residual, module.low_refine.residual):
                for parameter in residual.parameters():
                    parameter.add_(0.01)
        with torch.no_grad():
            expected = module(inputs)
        clone = ACFRConcat([64, 64])
        clone.load_state_dict(module.state_dict())
        with torch.no_grad():
            self.assertTrue(torch.equal(clone(inputs), expected))
        optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
        self.assertEqual(
            {id(p) for group in optimizer.param_groups for p in group["params"]},
            {id(p) for p in module.parameters()},
        )

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA device required")
    def test_cuda_amp_finite(self) -> None:
        """FP16 autocast forward/backward must stay finite on CUDA."""
        module = ACFRConcat([64, 64]).cuda()
        high = torch.randn(2, 64, 40, 44, device="cuda", requires_grad=True)
        low = torch.randn(2, 64, 40, 44, device="cuda", requires_grad=True)
        with torch.autocast("cuda", dtype=torch.float16):
            output = module([high, low])
        output.float().square().mean().backward()
        self.assertTrue(torch.isfinite(high.grad).all())
        self.assertTrue(torch.isfinite(low.grad).all())


class ACFRModelIntegrationTest(unittest.TestCase):
    """Verify the candidate YAML equals ACR except one ACFRConcat at L18."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.input = torch.randn(1, 3, 640, 640)
        cls.acr = DetectionModel(ACR_YAML, ch=3, nc=10, verbose=False).eval()
        cls.candidate = DetectionModel(CANDIDATE_YAML, ch=3, nc=10, verbose=False).eval()

    def test_single_module_and_widths(self) -> None:
        """Exactly one ACFRConcat; L18/L19/L28 widths and strides match ACR."""
        self.assertEqual(sum(isinstance(m, ACFRConcat) for m in self.candidate.modules()), 1)
        self.assertEqual(self.candidate.stride.tolist(), [4.0, 8.0, 16.0, 32.0])

        captured: dict[str, tuple[int, int]] = {}

        def record(model, index, label):
            def hook(_m, _i, output):
                tensor = output if torch.is_tensor(output) else output[0]
                captured[label] = (int(tensor.shape[1]), int(tensor.shape[-1]))
            return model.model[index].register_forward_hook(hook)

        handles = [
            record(self.acr, 18, "acr_l18"), record(self.candidate, 18, "cand_l18"),
            record(self.candidate, 19, "cand_l19"), record(self.candidate, 28, "cand_l28"),
        ]
        with torch.no_grad():
            self.acr(self.input.clone())
            self.candidate(self.input.clone())
        for handle in handles:
            handle.remove()
        self.assertEqual(captured["cand_l18"][0], 128)
        self.assertEqual(captured["acr_l18"], captured["cand_l18"])
        self.assertEqual(captured["cand_l19"][0], 64)
        self.assertEqual(captured["cand_l28"][0], 192)

    def test_parameter_boundary_and_delta(self) -> None:
        """Candidate adds parameters only under model.18.; every other key matches ACR."""
        acr_state = self.acr.state_dict()
        cand_state = self.candidate.state_dict()
        self.assertTrue(set(acr_state) <= set(cand_state))
        new_keys = set(cand_state) - set(acr_state)
        self.assertTrue(new_keys)
        self.assertTrue(all(key.startswith("model.18.") for key in new_keys))
        delta = sum(p.numel() for p in self.candidate.parameters()) - sum(
            p.numel() for p in self.acr.parameters()
        )
        self.assertGreater(delta, 0)
        self.assertLess(delta, 20_000)
        incompatible = self.candidate.load_state_dict(acr_state, strict=False)
        self.assertFalse(incompatible.unexpected_keys)
        self.assertTrue(all(key.startswith("model.18.") for key in incompatible.missing_keys))

    def test_no_closed_modules_and_eval_forward(self) -> None:
        """Closed-route modules must not appear; eval forward stays finite."""
        from sdf.modules import DWF, DWFNearestGate, DWFNearestWeight, HaarDownsample, SDD
        from sdf.modules import IdentityChannelAttentionConcat, ResidualDetailContextConcat

        closed = (DWF, DWFNearestGate, DWFNearestWeight, SDD, HaarDownsample,
                  IdentityChannelAttentionConcat, ResidualDetailContextConcat)
        for module_type in closed:
            self.assertEqual(sum(isinstance(m, module_type) for m in self.candidate.modules()), 0)
        with torch.no_grad():
            outputs = self.candidate(self.input.clone())
        tensor = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
        if isinstance(tensor, (list, tuple)):
            tensor = tensor[0]
        if tensor.ndim == 3:
            tensor = tensor[0]
        self.assertEqual(tensor.shape[1], 6)
        self.assertTrue(torch.isfinite(tensor).all())


if __name__ == "__main__":
    unittest.main()
