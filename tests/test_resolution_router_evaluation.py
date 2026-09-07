"""Tests for learned and baseline resolution-routing scores."""

from __future__ import annotations

import unittest

import numpy as np

from scripts.evaluate_resolution_router import coarse_small_scores, hash_random_scores


class ResolutionRouterEvaluationTest(unittest.TestCase):
    def test_coarse_score_uses_only_original_pixel_small_boxes(self) -> None:
        predictions = [
            {
                "boxes_xyxy": [[0, 0, 10, 10], [0, 0, 40, 40]],
                "scores": [0.25, 0.9],
            }
        ]
        np.testing.assert_allclose(coarse_small_scores(predictions), np.asarray([0.25]))

    def test_hash_random_scores_are_deterministic_and_seeded(self) -> None:
        images = ["a.jpg", "b.jpg"]
        np.testing.assert_array_equal(hash_random_scores(images, 0), hash_random_scores(images, 0))
        self.assertFalse(np.array_equal(hash_random_scores(images, 0), hash_random_scores(images, 1)))


if __name__ == "__main__":
    unittest.main()
