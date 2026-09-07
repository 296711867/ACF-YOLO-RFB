"""Apply the frozen P14 three-seed internal continuation gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from scripts.train_perspective_router import routing_metrics
from sdf.refinement import ensemble_lower_confidence_score


def read_scores(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", type=Path, nargs=3, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--route-fraction", type=float, default=0.25)
    args = parser.parse_args()
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.reports]
    if sorted(report["seed"] for report in reports) != [0, 1, 2]:
        raise ValueError("reports must contain exactly seeds 0, 1, and 2")
    score_sets = [read_scores(Path(report["scores"])) for report in reports]
    names = [item["image"] for item in score_sets[0]]
    if any([item["image"] for item in records] != names for records in score_sets[1:]):
        raise ValueError("seed holdout score order differs")
    utilities = np.asarray([item["utility"] for item in score_sets[0]], dtype=np.float32)
    if any(not np.array_equal(utilities, np.asarray([item["utility"] for item in records])) for records in score_sets[1:]):
        raise ValueError("seed holdout utilities differ")
    means = torch.tensor([[item["mean"] for item in records] for records in score_sets])
    log_variances = torch.tensor([[item["log_variance"] for item in records] for records in score_sets])
    ensemble_scores = ensemble_lower_confidence_score(means, log_variances).numpy()
    targets = (utilities > 0).astype(np.float32)
    ensemble_metrics = routing_metrics(ensemble_scores, utilities, targets, args.route_fraction)
    baselines = reports[0]["baselines"]
    learned_baselines = ("ridge_statistics", "hist_gradient_boosting")
    best_non_neural = max((name for name in baselines if name != "hash_random"), key=lambda name: baselines[name]["utility_capture"])
    fixed_heuristics = ("detection_count", "small_confidence_sum", "uncertainty_crowding")
    best_heuristic = max(fixed_heuristics, key=lambda name: baselines[name]["target_precision"])
    baseline_capture = baselines[best_non_neural]["utility_capture"]
    relative_gain = 100.0 * (ensemble_metrics["utility_capture"] / baseline_capture - 1.0) if baseline_capture else float("-inf")
    checks = {
        "all_seeds_not_below_best_non_neural": all(report["neural"]["utility_capture"] >= baseline_capture for report in reports),
        "ensemble_relative_gain_at_least_5_percent": relative_gain >= 5.0,
        "ensemble_positive_precision_above_best_fixed_heuristic": ensemble_metrics["target_precision"] > baselines[best_heuristic]["target_precision"],
        "reports_exclude_val_and_test": all("val and test-dev excluded" in report["status"] for report in reports),
    }
    summary = {
        "status": "continue to one-time val" if all(checks.values()) else "stop before val",
        "passed": all(checks.values()), "checks": checks,
        "seeds": {str(report["seed"]): report["neural"] for report in reports},
        "ensemble": ensemble_metrics, "baselines": baselines,
        "best_non_neural": best_non_neural, "best_fixed_heuristic": best_heuristic,
        "ensemble_relative_utility_capture_percent": relative_gain,
        "learned_baseline_names": list(learned_baselines),
        "inputs": {
            str(report["seed"]): {
                "report": str(path.resolve()), "report_sha256": sha256(path),
                "scores": report["scores"], "scores_sha256": sha256(Path(report["scores"])),
                "checkpoint": report["checkpoint"], "checkpoint_sha256": sha256(Path(report["checkpoint"])),
            }
            for path, report in zip(args.reports, reports)
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
