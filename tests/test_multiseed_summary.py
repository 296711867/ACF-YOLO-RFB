"""Tests for VisDrone multi-seed aggregation."""

from __future__ import annotations

import unittest

from scripts.summarize_visdrone_multiseed import cumulative_training_seconds, summarize


class MultiSeedSummaryTest(unittest.TestCase):
    """Verify sample statistics and resumed timer handling."""

    def test_summary_uses_sample_standard_deviation(self) -> None:
        result = summarize([1.0, 2.0, 3.0])
        self.assertEqual(result["mean"], 2.0)
        self.assertEqual(result["sample_std"], 1.0)
        self.assertEqual(result["coefficient_of_variation_percent"], 50.0)

    def test_cumulative_seconds_includes_resumed_segments(self) -> None:
        rows = [{"time": "20"}, {"time": "40"}, {"time": "10"}, {"time": "30"}]
        self.assertEqual(cumulative_training_seconds(rows), 70.0)


if __name__ == "__main__":
    unittest.main()
