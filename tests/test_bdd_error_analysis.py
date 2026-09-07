"""Synthetic tests for deterministic BDD qualitative-error analysis."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from scripts.analyze_bdd_errors import (
    Detections,
    ImageAnalysis,
    classify_events,
    event_sort_key,
    greedy_match,
    select_unique_images,
)


def detections(boxes, classes, scores=None) -> Detections:
    boxes_array = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    if scores is None:
        scores = np.ones(len(boxes_array), dtype=np.float32)
    return Detections(boxes_array, np.asarray(classes, dtype=np.int64), np.asarray(scores, dtype=np.float32))


class BDDErrorAnalysisTest(unittest.TestCase):
    def test_greedy_matching_is_class_aware_and_one_to_one(self) -> None:
        gt = detections([[0, 0, 10, 10], [0, 0, 10, 10]], [0, 1])
        pred = detections([[0, 0, 10, 10], [0, 0, 10, 10], [0, 0, 10, 10]], [0, 0, 1], [0.9, 0.8, 0.7])
        match = greedy_match(gt, pred, 0.5)
        self.assertEqual(match.pred_to_gt.tolist(), [0, -1, 1])
        self.assertEqual(match.gt_to_pred.tolist(), [0, 2])

    def test_all_three_event_types_and_fp_correspondence(self) -> None:
        gt = detections([[0, 0, 10, 10], [20, 0, 25, 5]], [0, 1])
        baseline = detections([[40, 0, 50, 10]], [2], [0.8])
        p2 = detections([[0, 0, 10, 10], [40, 0, 50, 10], [60, 0, 70, 10]], [0, 2, 2], [0.9, 0.85, 0.75])
        analysis = ImageAnalysis(
            Path("synthetic.jpg"), 100, 100, gt, baseline, p2, greedy_match(gt, baseline, 0.5), greedy_match(gt, p2, 0.5)
        )
        events = classify_events(analysis, 0.5, 1024.0)
        self.assertEqual(len(events["p2_repaired_small"]), 1)
        self.assertEqual(len(events["shared_small_miss"]), 1)
        self.assertEqual(len(events["p2_added_false_positive"]), 1)
        self.assertEqual(events["p2_added_false_positive"][0]["p2_box_xyxy"], [60.0, 0.0, 70.0, 10.0])

    def test_frozen_sorting_and_unique_image_selection(self) -> None:
        events = [
            {"image": "b.jpg", "p2_confidence": 0.8, "gt_box_xyxy": [0, 0, 2, 2], "class_id": 0},
            {"image": "a.jpg", "p2_confidence": 0.9, "gt_box_xyxy": [0, 0, 2, 2], "class_id": 0},
            {"image": "a.jpg", "p2_confidence": 0.7, "gt_box_xyxy": [3, 3, 5, 5], "class_id": 0},
        ]
        events.sort(key=lambda event: event_sort_key("p2_repaired_small", event))
        self.assertEqual([event["image"] for event in events], ["a.jpg", "b.jpg", "a.jpg"])
        self.assertEqual([event["image"] for event in select_unique_images(events, 2)], ["a.jpg", "b.jpg"])


if __name__ == "__main__":
    unittest.main()
