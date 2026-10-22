"""Regression tests for the P28 SimAM module."""

from __future__ import annotations

import unittest

import torch

from sdf.modules import SimAM


class TestSimAM(unittest.TestCase):
    def test_zero_parameters(self):
        module = SimAM()
        self.assertEqual(sum(p.numel() for p in module.parameters()), 0)

    def test_output_shape(self):
        module = SimAM()
        for shape in ((2, 64, 160, 160), (1, 3, 32, 32)):
            x = torch.randn(*shape)
            self.assertEqual(module(x).shape, x.shape)

    def test_output_bounded_scaling(self):
        # sigmoid weight in (0,1): output magnitude never exceeds input magnitude.
        module = SimAM()
        x = torch.randn(4, 8, 20, 20) * 10
        out = module(x)
        self.assertTrue(torch.all(out.abs() <= x.abs() + 1e-6))

    def test_deterministic_eval(self):
        module = SimAM().eval()
        x = torch.randn(2, 16, 40, 40)
        with torch.no_grad():
            self.assertTrue(torch.equal(module(x), module(x)))

    def test_gradient_flows(self):
        x = torch.randn(2, 8, 16, 16, requires_grad=True)
        SimAM()(x).sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertGreater(x.grad.abs().sum().item(), 0.0)

    def test_yaml_arg_rejection(self):
        from ultralytics import YOLO

        with self.assertRaises(Exception):
            YOLO("configs/models/yolo26n-p2-acr-acfr-simam-bad-args.yaml", task="detect")

    def test_model_zero_param_delta(self):
        from ultralytics import YOLO

        l0 = YOLO("configs/models/yolo26n-p2-acr-acfr.yaml", task="detect")
        s1 = YOLO("configs/models/yolo26n-p2-acr-acfr-simam.yaml", task="detect")
        p_l0 = sum(p.numel() for p in l0.model.parameters())
        p_s1 = sum(p.numel() for p in s1.model.parameters())
        self.assertEqual(p_s1 - p_l0, 0)
        self.assertEqual(len(s1.model.model) - len(l0.model.model), 1)
        self.assertEqual(type(s1.model.model[20]).__name__, "SimAM")


if __name__ == "__main__":
    unittest.main()
