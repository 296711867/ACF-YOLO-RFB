"""Tests for sparse high-resolution refinement primitives and manifests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from sahi.prediction import ObjectPrediction

from scripts.create_refinement_dev_manifest import ranked_names
from scripts.run_sparse_refinement_feasibility import load_manifest
from sdf.refinement import (
    greedy_oracle_tiles,
    merge_object_predictions,
    merge_scale_aware_predictions,
    object_predictions_to_arrays,
    shift_and_filter_local_predictions,
)


class SparseRefinementTest(unittest.TestCase):
    def test_oracle_greedily_maximizes_unique_center_coverage(self) -> None:
        tiles = [[0, 0, 60, 100], [40, 0, 100, 100], [80, 0, 140, 100]]
        targets = np.asarray([[5, 5, 15, 15], [45, 5, 55, 15], [105, 5, 115, 15]], dtype=np.float32)
        selected, coverage = greedy_oracle_tiles(tiles, targets, top_k=2)
        self.assertEqual(selected, [0, 2])
        self.assertEqual(coverage, 1.0)

    def test_class_aware_fusion_keeps_different_classes(self) -> None:
        first = ObjectPrediction(bbox=[0, 0, 20, 20], category_id=1, category_name="rider", score=0.9)
        duplicate = ObjectPrediction(bbox=[1, 1, 19, 19], category_id=1, category_name="rider", score=0.8)
        other_class = ObjectPrediction(bbox=[1, 1, 19, 19], category_id=2, category_name="car", score=0.7)
        merged = merge_object_predictions([first, duplicate, other_class])
        boxes, classes, scores = object_predictions_to_arrays(merged)
        self.assertEqual(boxes.shape, (2, 4))
        np.testing.assert_array_equal(np.sort(classes), np.asarray([1, 2]))
        self.assertAlmostEqual(float(scores.max()), 0.9)

    def test_local_predictions_are_shifted_and_area_filtered(self) -> None:
        small = ObjectPrediction(
            bbox=[1, 2, 11, 12], category_id=2, category_name="car", score=0.9, shift_amount=[100, 200]
        )
        large = ObjectPrediction(
            bbox=[0, 0, 40, 40], category_id=2, category_name="car", score=0.8, shift_amount=[100, 200]
        )
        selected = shift_and_filter_local_predictions([small, large], max_area=32**2)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].bbox.to_xyxy(), [101, 202, 111, 212])

    def test_scale_aware_fusion_preserves_full_frame_large_boxes(self) -> None:
        large = ObjectPrediction(bbox=[0, 0, 40, 40], category_id=2, category_name="car", score=0.6)
        small_inside = ObjectPrediction(bbox=[5, 5, 15, 15], category_id=2, category_name="car", score=0.9)
        fused = merge_scale_aware_predictions([large], [small_inside], max_local_area=32**2)
        boxes, _, _ = object_predictions_to_arrays(fused)
        self.assertEqual(len(fused), 2)
        self.assertIn([0.0, 0.0, 40.0, 40.0], boxes.tolist())

    def test_manifest_selection_and_loading_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("c.jpg", "a.jpg", "b.jpg"):
                (root / name).touch()
            ranked = ranked_names(root, 7)
            self.assertEqual(ranked, ranked_names(root, 7))
            self.assertTrue(set(ranked[:1]).isdisjoint(ranked[1:]))
            manifest = root / "manifest.txt"
            manifest.write_text("b.jpg\na.jpg\n", encoding="utf-8")
            self.assertEqual([path.name for path in load_manifest(manifest, root)], ["b.jpg", "a.jpg"])


if __name__ == "__main__":
    unittest.main()
