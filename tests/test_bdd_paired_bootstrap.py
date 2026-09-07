"""Tests for paired BDD image-bootstrap AP helpers."""

from __future__ import annotations

import unittest

import numpy as np

from scripts.bootstrap_bdd_paired_ap import PreparedModel, RankedDetections, model_ap, weighted_ap


class BDDPairedBootstrapTest(unittest.TestCase):
    """Verify weighted image replication without requiring full BDD evidence."""

    def test_zero_weight_image_is_excluded(self) -> None:
        ranked = RankedDetections(
            image_indices=np.asarray([0, 1], dtype=np.int32),
            true_positive=np.asarray([False, True]),
        )
        self.assertAlmostEqual(weighted_ap(ranked, np.asarray([0.0, 1.0]), positives=1.0), 1.0)

    def test_repeated_high_score_false_positive_reduces_ap(self) -> None:
        ranked = RankedDetections(
            image_indices=np.asarray([0, 1], dtype=np.int32),
            true_positive=np.asarray([False, True]),
        )
        once = weighted_ap(ranked, np.asarray([1.0, 1.0]), positives=1.0)
        repeated = weighted_ap(ranked, np.asarray([2.0, 1.0]), positives=1.0)
        self.assertLess(repeated, once)

    def test_missing_resampled_class_keeps_fixed_macro_denominator(self) -> None:
        empty = RankedDetections(np.asarray([], dtype=np.int32), np.asarray([], dtype=bool))
        model = PreparedModel(
            positives={("all", class_id): np.asarray([1 if class_id == 0 else 0]) for class_id in range(10)},
            detections={("all", class_id, threshold): empty for class_id in range(10) for threshold in range(10)},
        )
        self.assertEqual(model_ap(model, np.asarray([0.0]), "all"), 0.0)


if __name__ == "__main__":
    unittest.main()
