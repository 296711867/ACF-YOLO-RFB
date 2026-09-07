"""Tests for integrated adaptive-resolution benchmark protocol helpers."""

from __future__ import annotations

import unittest

from pathlib import Path

import numpy as np

from scripts.benchmark_integrated_router import POLICIES, path_batches, policy_order, selection_sha256, summarize_runs


class IntegratedRouterBenchmarkTest(unittest.TestCase):
    def test_path_batches_enforces_requested_batch(self) -> None:
        paths = [Path(f"{index}.jpg") for index in range(35)]
        batches = path_batches(paths, 16)
        self.assertEqual([len(batch) for batch in batches], [16, 16, 3])
        self.assertEqual([path for batch in batches for path in batch], paths)
        with self.assertRaises(ValueError):
            path_batches(paths, 0)

    def test_policy_order_rotates_without_changing_membership(self) -> None:
        self.assertEqual(policy_order(0), POLICIES)
        self.assertEqual(policy_order(1)[0], POLICIES[1])
        for repeat in range(len(POLICIES)):
            self.assertEqual(set(policy_order(repeat)), set(POLICIES))

    def test_summary_uses_sample_standard_deviation(self) -> None:
        runs = []
        for value in (1.0, 2.0, 3.0):
            runs.append(
                {
                    "total_seconds": value,
                    "coarse_seconds": value,
                    "routing_seconds": 0.0,
                    "refinement_seconds": 0.0,
                    "throughput_images_per_second": 10.0 / value,
                    "routed_images": 5,
                    "selection_sha256": "same-selection",
                    "peak_allocated_mib": 100.0,
                    "peak_reserved_mib": 120.0,
                }
            )
        summary = summarize_runs(runs)
        self.assertEqual(summary["total_seconds"]["mean"], 2.0)
        self.assertEqual(summary["total_seconds"]["sample_std"], 1.0)
        self.assertEqual(summary["routed_images"], [5])
        self.assertEqual(summary["selection_sha256"], ["same-selection"])

    def test_selection_hash_tracks_membership_not_score_order(self) -> None:
        paths = [Path("a.jpg"), Path("b.jpg"), Path("c.jpg")]
        first = selection_sha256(paths, np.asarray([2, 0]))
        self.assertEqual(first, selection_sha256(paths, np.asarray([0, 2])))
        self.assertNotEqual(first, selection_sha256(paths, np.asarray([0, 1])))


if __name__ == "__main__":
    unittest.main()
