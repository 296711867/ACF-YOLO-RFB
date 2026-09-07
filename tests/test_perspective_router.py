"""Tests for the learnable perspective-band resolution router."""

from __future__ import annotations

import unittest

import torch

from sdf.refinement import PerspectiveBandRouter, perspective_band_descriptor


class PerspectiveBandRouterTest(unittest.TestCase):
    def test_descriptor_preserves_vertical_band_order_and_statistics(self) -> None:
        feature = torch.zeros(1, 2, 4, 2)
        for row in range(4):
            feature[:, :, row] = row + 1
        descriptor = perspective_band_descriptor(feature, bands=4)
        self.assertEqual(descriptor.shape, (1, 16))
        self.assertEqual(descriptor[0, 0].item(), 1.0)
        self.assertEqual(descriptor[0, 4].item(), 2.0)
        self.assertEqual(descriptor[0, 12].item(), 4.0)

    def test_router_has_finite_gradients_and_rejects_wrong_channels(self) -> None:
        router = PerspectiveBandRouter(channels=2, bands=4, hidden=4)
        feature = torch.randn(3, 2, 8, 8, requires_grad=True)
        logits = router(feature)
        self.assertEqual(logits.shape, (3,))
        logits.sum().backward()
        self.assertTrue(torch.isfinite(feature.grad).all())
        self.assertTrue(all(parameter.grad is not None for parameter in router.parameters()))
        with self.assertRaises(ValueError):
            router(torch.randn(1, 3, 8, 8))


if __name__ == "__main__":
    unittest.main()
