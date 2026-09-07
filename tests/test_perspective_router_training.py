"""Tests for fixed-budget perspective-router training helpers."""

from __future__ import annotations

import unittest

import numpy as np

from scripts.train_perspective_router import routing_metrics, top_fraction_indices


class PerspectiveRouterTrainingTest(unittest.TestCase):
    def test_top_fraction_is_stable_and_uses_exact_ceiling_budget(self) -> None:
        selected = top_fraction_indices(np.asarray([0.5, 0.5, 0.1, 0.4]), 0.25)
        np.testing.assert_array_equal(selected, np.asarray([0]))
        selected = top_fraction_indices(np.asarray([0.5, 0.5, 0.1]), 0.5)
        np.testing.assert_array_equal(selected, np.asarray([0, 1]))

    def test_routing_metrics_reports_full_oracle_capture(self) -> None:
        utilities = np.asarray([3.0, 2.0, -1.0, 0.0])
        targets = np.asarray([1.0, 1.0, 0.0, 0.0])
        metrics = routing_metrics(utilities, utilities, targets, 0.5)
        self.assertEqual(metrics["target_precision"], 1.0)
        self.assertEqual(metrics["utility_capture"], 1.0)


if __name__ == "__main__":
    unittest.main()
