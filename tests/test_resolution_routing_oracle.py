"""Tests for the adaptive-resolution routing oracle."""

from __future__ import annotations

import unittest

import numpy as np

from scripts.analyze_resolution_routing_oracle import select_from_utilities, select_oracle_routes, small_match_utility
from scripts.evaluate_object_sizes import ImageDetections


def record(score: float, box: list[float]) -> ImageDetections:
    gt_box = np.asarray([[0, 0, 10, 10]], dtype=np.float32)
    pred_box = np.asarray([box], dtype=np.float32)
    return ImageDetections(
        gt_boxes=gt_box,
        gt_classes=np.asarray([0]),
        gt_areas=np.asarray([100], dtype=np.float32),
        pred_boxes=pred_box,
        pred_classes=np.asarray([0]),
        pred_scores=np.asarray([score], dtype=np.float32),
        pred_areas=np.asarray([(box[2] - box[0]) * (box[3] - box[1])], dtype=np.float32),
    )


class ResolutionRoutingOracleTest(unittest.TestCase):
    def test_utility_is_positive_when_high_resolution_recovers_small_match(self) -> None:
        baseline = record(0.9, [20, 20, 30, 30])
        high_res = record(0.9, [0, 0, 10, 10])
        self.assertEqual(small_match_utility(baseline, high_res), 10)

    def test_fixed_budget_selects_highest_utility_with_stable_ties(self) -> None:
        missed = record(0.9, [20, 20, 30, 30])
        found = record(0.9, [0, 0, 10, 10])
        selected, utilities = select_oracle_routes([missed, found, missed], [found, found, missed], fraction=1 / 3)
        self.assertEqual(utilities, [10, 0, 0])
        self.assertEqual(selected, {0})

    def test_fraction_selection_rejects_invalid_budget(self) -> None:
        with self.assertRaises(ValueError):
            select_from_utilities([1, 0], 1.1)


if __name__ == "__main__":
    unittest.main()
