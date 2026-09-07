"""Tests for the P14 scout detection-set router and cache encoding."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from scripts.prepare_scout_detection_cache import encode_detection_tokens, sequence_id
from sdf.refinement import ScoutDetectionValueRouter, ensemble_lower_confidence_score, scout_value_loss


class ScoutDetectionValueRouterTest(unittest.TestCase):
    def test_permutation_invariance_gradient_and_parameter_budget(self) -> None:
        torch.manual_seed(0)
        model = ScoutDetectionValueRouter()
        tokens = torch.rand(3, 8, 7, requires_grad=True)
        classes = torch.randint(0, 10, (3, 8))
        mask = torch.tensor([[True] * 8, [True] * 5 + [False] * 3, [False] * 8])
        means, log_variances = model(tokens, classes, mask)
        order = torch.tensor([3, 0, 7, 2, 6, 1, 5, 4])
        permuted = model(tokens[:, order], classes[:, order], mask[:, order])
        self.assertTrue(torch.allclose(means, permuted[0], atol=1e-6))
        self.assertTrue(torch.allclose(log_variances, permuted[1], atol=1e-6))
        self.assertTrue(torch.isfinite(means).all())
        self.assertTrue(torch.isfinite(log_variances).all())
        self.assertLessEqual(sum(parameter.numel() for parameter in model.parameters()), 50_000)
        loss, components = scout_value_loss(means, log_variances, torch.tensor([3.0, 0.0, -2.0]), 0.0, 2.0)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(value) for value in components.values()))
        self.assertTrue(torch.isfinite(tokens.grad).all())

    def test_ensemble_score_penalizes_total_uncertainty(self) -> None:
        means = torch.tensor([[2.0, 2.0], [2.0, 4.0], [2.0, 0.0]])
        log_variances = torch.zeros_like(means)
        scores = ensemble_lower_confidence_score(means, log_variances)
        self.assertGreater(scores[0], scores[1])
        with self.assertRaises(ValueError):
            ensemble_lower_confidence_score(means[:1], log_variances[:1])

    def test_detection_encoding_is_stable_normalized_and_padded(self) -> None:
        prediction = {
            "width": 100,
            "height": 50,
            "boxes_xyxy": [[0, 0, 10, 10], [50, 10, 100, 50]],
            "classes": [2, 3],
            "scores": [0.2, 0.9],
        }
        tokens, classes, mask = encode_detection_tokens(prediction, max_tokens=3)
        self.assertEqual(classes.tolist(), [3, 2, 0])
        self.assertEqual(mask.tolist(), [True, True, False])
        self.assertAlmostEqual(float(tokens[0, 0]), 0.75)
        self.assertAlmostEqual(float(tokens[0, 4]), 0.9)
        self.assertTrue(np.isfinite(tokens).all())
        self.assertTrue(np.equal(tokens[2], 0).all())
        self.assertEqual(sequence_id("0000010_05149_d_0000057.jpg"), "0000010")

    def test_invalid_shapes_and_classes_are_rejected(self) -> None:
        model = ScoutDetectionValueRouter()
        with self.assertRaises(ValueError):
            model(torch.zeros(2, 4, 6), torch.zeros(2, 4, dtype=torch.long), torch.ones(2, 4, dtype=torch.bool))
        with self.assertRaises(ValueError):
            encode_detection_tokens(
                {"width": 10, "height": 10, "boxes_xyxy": [[0, 0, 1, 1]], "classes": [10], "scores": [0.5]}
            )


if __name__ == "__main__":
    unittest.main()
