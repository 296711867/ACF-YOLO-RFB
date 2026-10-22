"""Component-ablation variants of ACFRConcat: gate-only and refine-only.

These tests guarantee (1) both variants stay identity-initialized, (2) the
default constructor path is unchanged by the ablation switches, and (3) the
variant YAMLs build with the expected parameter deltas over the frozen ACR
structure.
"""

from __future__ import annotations

import unittest

import torch
from ultralytics import YOLO

from sdf.modules.acfr import ACFRConcat

CHANNELS = [64, 64]


def _inputs(seed: int = 0) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return [torch.randn(2, 64, 20, 20, generator=generator) for _ in range(2)]


class ACFRVariantModuleTest(unittest.TestCase):
    def test_refine_only_identity_and_params(self) -> None:
        module = ACFRConcat(CHANNELS, use_gate=False, use_refine=True)
        names = [name for name, _ in module.named_parameters()]
        self.assertTrue(all(".gate" not in name for name in names))
        self.assertTrue(any(name.startswith("high_refine") for name in names))
        self.assertTrue(any(name.startswith("low_refine") for name in names))
        with torch.no_grad():
            output = module(_inputs())
        expected = torch.cat(_inputs(), dim=1)
        self.assertTrue(torch.equal(output, expected))
        self.assertEqual(output.shape, (2, 128, 20, 20))

    def test_gate_only_identity_and_params(self) -> None:
        module = ACFRConcat(CHANNELS, use_gate=True, use_refine=False)
        names = [name for name, _ in module.named_parameters()]
        self.assertTrue(all("refine" not in name for name in names))
        self.assertTrue(any(name.startswith("gate") for name in names))
        with torch.no_grad():
            output = module(_inputs())
        expected = torch.cat(_inputs(), dim=1)
        self.assertTrue(torch.equal(output, expected))

    def test_both_disabled_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ACFRConcat(CHANNELS, use_gate=False, use_refine=False)

    def test_variant_backward_and_optimizer_coverage(self) -> None:
        for use_gate, use_refine in ((False, True), (True, False)):
            module = ACFRConcat(CHANNELS, use_gate=use_gate, use_refine=use_refine)
            output = module(_inputs(seed=3))
            output.sum().backward()
            optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
            optimizer.step()
            for name, parameter in module.named_parameters():
                self.assertTrue(parameter.requires_grad, name)

    def test_default_path_unchanged_by_switches(self) -> None:
        torch.manual_seed(11)
        default = ACFRConcat(CHANNELS)
        torch.manual_seed(11)
        explicit = ACFRConcat(CHANNELS, use_gate=True, use_refine=True)
        default_state = default.state_dict()
        explicit_state = explicit.state_dict()
        self.assertEqual(sorted(default_state), sorted(explicit_state))
        for key in default_state:
            self.assertTrue(torch.equal(default_state[key], explicit_state[key]), key)
        inputs = _inputs(seed=7)
        with torch.no_grad():
            self.assertTrue(torch.equal(default(inputs), explicit(inputs)))


class ACFRVariantModelTest(unittest.TestCase):
    def test_variant_yamls_build_with_expected_deltas(self) -> None:
        base = YOLO("configs/models/yolo26n-p2-acr.yaml")
        base_params = sum(p.numel() for p in base.model.parameters())
        for yaml_name, expected_delta, expected_gate in (
            ("yolo26n-p2-acr-acfr-nogate.yaml", 2816, False),
            ("yolo26n-p2-acr-acfr-norefine.yaml", 5316, True),
        ):
            model = YOLO(f"configs/models/{yaml_name}")
            layer18 = model.model.model[18]
            self.assertIsInstance(layer18, ACFRConcat)
            self.assertEqual(layer18.use_gate, expected_gate)
            self.assertNotEqual(layer18.use_refine, expected_gate)
            params = sum(p.numel() for p in model.model.parameters())
            self.assertEqual(params - base_params, expected_delta)


if __name__ == "__main__":
    unittest.main()
