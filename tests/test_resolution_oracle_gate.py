"""Tests for the frozen P13 continuation gate."""

from __future__ import annotations

import unittest

from scripts.evaluate_resolution_oracle_gate import evaluate_gate


def report(all_ap: float, small_ap: float, small_ar: float) -> dict:
    return {"metrics": {"all": {"AP50_95": all_ap}, "small": {"AP50_95": small_ap, "AR50_95": small_ar}}}


class ResolutionOracleGateTest(unittest.TestCase):
    def test_passes_only_when_all_frozen_conditions_hold(self) -> None:
        baseline = report(0.20, 0.10, 0.20)
        high_res = report(0.20, 0.12, 0.23)
        oracle = {"budgets": {"0.25": {"metrics": report(0.20, 0.111, 0.21)["metrics"]}}}
        self.assertTrue(evaluate_gate(baseline, high_res, oracle)["passed"])

    def test_fails_when_oracle_does_not_retain_half_small_ap_gain(self) -> None:
        baseline = report(0.20, 0.10, 0.20)
        high_res = report(0.20, 0.12, 0.23)
        oracle = {"budgets": {"0.25": {"metrics": report(0.20, 0.109, 0.21)["metrics"]}}}
        result = evaluate_gate(baseline, high_res, oracle)
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["oracle25_retains_half_full960_small_ap_gain"])


if __name__ == "__main__":
    unittest.main()
