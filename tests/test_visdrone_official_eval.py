"""Tests for VisDrone official-format export and evaluator compatibility."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.evaluate_visdrone_official import evaluate
from scripts.export_visdrone_predictions import format_detections


class VisDroneOfficialEvaluationTest(unittest.TestCase):
    """Exercise format conversion, official matching, and ignored regions."""

    def test_export_sorts_limits_and_maps_classes(self) -> None:
        """Rows must be score-sorted, use XYWH, and convert classes to one-based IDs."""
        text = format_detections(
            np.asarray([[1, 2, 11, 22], [5, 6, 8, 10], [0, 0, 4, 4]], dtype=float),
            np.asarray([0.2, 0.9, 0.8]),
            np.asarray([0, 9, 3]),
            max_det=2,
        )
        rows = text.strip().splitlines()
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].split(",")[5:], ["10", "-1", "-1"])
        self.assertEqual(rows[1].split(",")[5:], ["4", "-1", "-1"])
        self.assertEqual(rows[0].split(",")[:4], ["5.000000", "6.000000", "3.000000", "4.000000"])

    def test_perfect_prediction_scores_one_hundred(self) -> None:
        """One exact class-1 prediction must produce 100 percent AP and AR."""
        result = self._evaluate(
            ground_truth="10,10,20,20,1,1,0,0\n",
            detections="10,10,20,20,0.9,1,-1,-1\n",
        )
        for key in ("AP50_95", "AP50", "AP75", "AR1", "AR10", "AR100", "AR500"):
            self.assertAlmostEqual(result["metrics"][key], 100.0)

    def test_higher_scoring_false_positive_halves_ap(self) -> None:
        """A false positive ranked before one true positive yields 50 percent AP."""
        result = self._evaluate(
            ground_truth="10,10,20,20,1,1,0,0\n",
            detections="60,60,10,10,0.9,1,-1,-1\n10,10,20,20,0.8,1,-1,-1\n",
        )
        self.assertAlmostEqual(result["metrics"]["AP50_95"], 50.0)
        self.assertAlmostEqual(result["metrics"]["AR1"], 0.0)
        self.assertAlmostEqual(result["metrics"]["AR10"], 100.0)

    def test_detection_inside_ignored_region_is_removed(self) -> None:
        """A high-score false detection inside a class-0 ignored region must not reduce AP."""
        result = self._evaluate(
            ground_truth="0,0,50,50,0,0,0,0\n60,60,20,20,1,1,0,0\n",
            detections="5,5,20,20,0.9,1,-1,-1\n60,60,20,20,0.8,1,-1,-1\n",
        )
        self.assertAlmostEqual(result["metrics"]["AP50_95"], 100.0)

    @staticmethod
    def _evaluate(ground_truth: str, detections: str) -> dict:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotations = root / "annotations"
            predictions = root / "predictions"
            images = root / "images"
            annotations.mkdir()
            predictions.mkdir()
            images.mkdir()
            (annotations / "sample.txt").write_text(ground_truth, encoding="ascii")
            (predictions / "sample.txt").write_text(detections, encoding="ascii")
            Image.new("RGB", (100, 100)).save(images / "sample.jpg")
            return evaluate(annotations, predictions, images)


if __name__ == "__main__":
    unittest.main()
