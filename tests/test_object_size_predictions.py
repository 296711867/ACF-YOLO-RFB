"""Tests for resumable object-size evaluation predictions."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.evaluate_object_sizes import load_predictions


class ObjectSizePredictionTest(unittest.TestCase):
    """Verify completed prediction JSONL files can be evaluated without inference."""

    def test_load_predictions_rebuilds_ground_truth_and_prediction_areas(self) -> None:
        """The recovery path must preserve boxes, classes, scores, and derived areas."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "images/val/example.jpg"
            label = root / "labels/val/example.txt"
            image.parent.mkdir(parents=True)
            label.parent.mkdir(parents=True)
            image.touch()
            label.write_text("2 0.5 0.5 0.25 0.5\n", encoding="utf-8")
            predictions = root / "predictions.jsonl"
            predictions.write_text(
                json.dumps(
                    {
                        "image": str(image),
                        "width": 200,
                        "height": 100,
                        "boxes_xyxy": [[50.0, 25.0, 100.0, 75.0]],
                        "classes": [2],
                        "scores": [0.75],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            records = load_predictions(predictions)
            self.assertEqual(len(records), 1)
            record = records[0]
            np.testing.assert_array_equal(record.gt_classes, np.array([2]))
            np.testing.assert_allclose(record.gt_areas, np.array([2500.0]))
            np.testing.assert_allclose(record.pred_areas, np.array([2500.0]))
            np.testing.assert_allclose(record.pred_scores, np.array([0.75]))


if __name__ == "__main__":
    unittest.main()
