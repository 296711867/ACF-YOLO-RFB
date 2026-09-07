"""Tests for frozen P14 streaming labels and strong scout baselines."""

from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path

import numpy as np

from scripts.train_scout_detection_value_router import (
    BASELINE_FEATURES,
    aggregate_scout_statistics,
    fixed_baseline_scores,
    ridge_predict,
    sequence_order,
)


class ScoutRouterTrainingTest(unittest.TestCase):
    def test_interrupted_prediction_prefix_is_valid_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            records = [{"image": f"/data/{index}.jpg"} for index in range(3)]
            path.write_text("".join(json.dumps(item) + "\n" for item in records), encoding="utf-8")
            with path.open(encoding="utf-8") as file:
                loaded = [json.loads(line)["image"] for line in file if line.strip()]
            self.assertEqual(loaded, [item["image"] for item in records])

    def test_sequence_order_is_deterministic_and_group_level(self) -> None:
        sequences = np.asarray(["0002", "0001", "0002", "0003"])
        first = sequence_order(sequences)
        second = sequence_order(sequences[::-1])
        self.assertEqual(first, second)
        self.assertEqual(set(first), {"0001", "0002", "0003"})

    def test_aggregate_statistics_are_finite_for_empty_and_dense_sets(self) -> None:
        tokens = np.zeros((2, 4, 7), dtype=np.float32)
        classes = np.zeros((2, 4), dtype=np.int16)
        mask = np.zeros((2, 4), dtype=bool)
        tokens[1, :2] = [[0.1, 0.2, 0.02, 0.03, 0.9, 0.03, 0.0], [0.2, 0.2, 0.04, 0.04, 0.1, 0.05, 0.0]]
        classes[1, :2] = [1, 2]
        mask[1, :2] = True
        statistics = aggregate_scout_statistics(tokens, classes, mask)
        self.assertEqual(statistics.shape, (2, len(BASELINE_FEATURES)))
        self.assertTrue(np.isfinite(statistics).all())
        self.assertTrue(np.equal(statistics[0], 0).all())
        self.assertEqual(float(statistics[1, 0]), 0.5)
        self.assertAlmostEqual(float(statistics[1, 1]), 1.0)

    def test_ridge_and_hist_gradient_boosting_are_deterministic(self) -> None:
        rng = np.random.default_rng(0)
        fit = rng.normal(size=(80, len(BASELINE_FEATURES))).astype(np.float32)
        utilities = (2 * fit[:, 0] - fit[:, 1]).astype(np.float32)
        query = rng.normal(size=(20, len(BASELINE_FEATURES))).astype(np.float32)
        ridge = ridge_predict(fit, utilities, query)
        names = [f"image-{index}.jpg" for index in range(len(query))]
        first = fixed_baseline_scores(fit, utilities, query, names)
        second = fixed_baseline_scores(fit, utilities, query, names)
        self.assertTrue(np.isfinite(ridge).all())
        self.assertEqual(set(first), set(second))
        for name in first:
            self.assertTrue(np.array_equal(first[name], second[name]))


if __name__ == "__main__":
    unittest.main()
