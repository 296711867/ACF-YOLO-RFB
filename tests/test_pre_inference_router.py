"""Tests for the detector-free pre-inference utility router."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from scripts.prepare_thumbnail_router_cache import STATISTIC_NAMES, thumbnail_statistics
from scripts.train_thumbnail_utility_router import ridge_scores, top_utility_targets
from sdf.refinement import (
    ThumbnailUtilityRouter,
    thumbnail_utility_descriptor,
    utility_routing_loss,
    weighted_pairwise_ranking_loss,
)


class PreInferenceUtilityRouterTest(unittest.TestCase):
    """Keep the thumbnail model, loss, and same-cost baselines deterministic."""

    def test_model_shape_gradient_parameter_budget_and_serialization(self) -> None:
        torch.manual_seed(0)
        model = ThumbnailUtilityRouter()
        inputs = torch.rand(3, 3, 96, 160, requires_grad=True)
        utilities = torch.tensor([5.0, 0.0, -3.0])
        scores = model(inputs)
        self.assertEqual(scores.shape, (3,))
        self.assertLessEqual(sum(parameter.numel() for parameter in model.parameters()), 100_000)
        loss, components = utility_routing_loss(scores, utilities, utility_mean=1.0, utility_std=3.0)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(value) for value in components.values()))
        self.assertTrue(torch.isfinite(inputs.grad).all())
        self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters()))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "router.pt"
            torch.save(model.state_dict(), path)
            restored = ThumbnailUtilityRouter()
            restored.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            model.eval()
            restored.eval()
            with torch.inference_mode():
                self.assertTrue(torch.equal(model(inputs.detach()), restored(inputs.detach())))

    def test_descriptor_preserves_band_and_global_statistics(self) -> None:
        feature = torch.zeros(1, 2, 12, 20)
        feature[:, :, :3] = 4.0
        descriptor = thumbnail_utility_descriptor(feature, bands=4)
        self.assertEqual(descriptor.shape, (1, 20))
        self.assertTrue(torch.equal(descriptor[0, :2], torch.tensor([4.0, 4.0])))
        self.assertTrue(torch.equal(descriptor[0, -4:], torch.tensor([1.0, 1.0, 4.0, 4.0])))

    def test_pairwise_loss_rewards_correct_continuous_order(self) -> None:
        utilities = torch.tensor([4.0, 1.0, -2.0])
        correct = weighted_pairwise_ranking_loss(torch.tensor([3.0, 0.0, -3.0]), utilities)
        reversed_order = weighted_pairwise_ranking_loss(torch.tensor([-3.0, 0.0, 3.0]), utilities)
        self.assertLess(correct, reversed_order)
        equal_scores = torch.zeros(3, requires_grad=True)
        zero = weighted_pairwise_ranking_loss(equal_scores, torch.ones(3))
        zero.backward()
        self.assertEqual(float(zero), 0.0)
        self.assertTrue(torch.equal(equal_scores.grad, torch.zeros(3)))

    def test_thumbnail_statistics_and_ridge_baseline_are_finite(self) -> None:
        image = np.zeros((96, 160, 3), dtype=np.uint8)
        image[:, 80:] = 255
        statistics = thumbnail_statistics(image)
        self.assertEqual(statistics.shape, (len(STATISTIC_NAMES),))
        self.assertTrue(np.isfinite(statistics).all())
        self.assertGreater(statistics[2], 0.0)

        fit = np.asarray([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [3.0, 1.0]], dtype=np.float32)
        utilities = np.asarray([0.0, 2.0, 4.0, 6.0], dtype=np.float32)
        predictions = ridge_scores(fit, utilities, np.asarray([[4.0, 1.0], [5.0, 1.0]], dtype=np.float32))
        self.assertGreater(predictions[1], predictions[0])
        self.assertTrue(np.isfinite(predictions).all())

    def test_top_utility_targets_uses_exact_ceiling_budget(self) -> None:
        targets = top_utility_targets(np.asarray([2.0, 4.0, 1.0, 3.0, 0.0]), 0.25)
        self.assertEqual(int(targets.sum()), 2)
        self.assertEqual(targets.tolist(), [0.0, 1.0, 0.0, 1.0, 0.0])

    def test_invalid_input_and_loss_shapes_are_rejected(self) -> None:
        model = ThumbnailUtilityRouter()
        with self.assertRaises(ValueError):
            model(torch.rand(2, 1, 96, 160))
        with self.assertRaises(ValueError):
            weighted_pairwise_ranking_loss(torch.zeros(2, 1), torch.zeros(2))
        with self.assertRaises(ValueError):
            utility_routing_loss(torch.zeros(2), torch.zeros(2), 0.0, 0.0)


if __name__ == "__main__":
    unittest.main()
