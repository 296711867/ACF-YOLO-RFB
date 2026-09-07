"""Apply the frozen P13 VisDrone resolution-oracle continuation gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def relative_change(candidate: float, baseline: float) -> float:
    if baseline == 0:
        raise ValueError("baseline metric must be non-zero")
    return 100.0 * (candidate - baseline) / baseline


def evaluate_gate(baseline: dict, high_res: dict, oracle: dict) -> dict:
    b_all = float(baseline["metrics"]["all"]["AP50_95"])
    b_small_ap = float(baseline["metrics"]["small"]["AP50_95"])
    b_small_ar = float(baseline["metrics"]["small"]["AR50_95"])
    h_all = float(high_res["metrics"]["all"]["AP50_95"])
    h_small_ap = float(high_res["metrics"]["small"]["AP50_95"])
    h_small_ar = float(high_res["metrics"]["small"]["AR50_95"])
    oracle_metrics = oracle["budgets"]["0.25"]["metrics"]
    o_all = float(oracle_metrics["all"]["AP50_95"])
    o_small_ap = float(oracle_metrics["small"]["AP50_95"])
    o_small_ar = float(oracle_metrics["small"]["AR50_95"])
    full_small_gain = h_small_ap - b_small_ap
    retention = (o_small_ap - b_small_ap) / full_small_gain if full_small_gain > 0 else float("-inf")
    checks = {
        "full960_small_ap_relative_gain_at_least_5_percent": relative_change(h_small_ap, b_small_ap) >= 5.0,
        "full960_small_ar_strictly_positive": h_small_ar > b_small_ar,
        "full960_all_ap_not_below_minus_2_percent": relative_change(h_all, b_all) >= -2.0,
        "oracle25_small_ap_above_baseline": o_small_ap > b_small_ap,
        "oracle25_small_ar_above_baseline": o_small_ar > b_small_ar,
        "oracle25_retains_half_full960_small_ap_gain": retention >= 0.5,
        "oracle25_all_ap_not_below_minus_2_percent": relative_change(o_all, b_all) >= -2.0,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": {
            "baseline": {"all_ap": b_all, "small_ap": b_small_ap, "small_ar": b_small_ar},
            "full960": {"all_ap": h_all, "small_ap": h_small_ap, "small_ar": h_small_ar},
            "oracle25": {"all_ap": o_all, "small_ap": o_small_ap, "small_ar": o_small_ar},
        },
        "changes": {
            "full960_small_ap_relative_percent": relative_change(h_small_ap, b_small_ap),
            "full960_small_ar_relative_percent": relative_change(h_small_ar, b_small_ar),
            "full960_all_ap_relative_percent": relative_change(h_all, b_all),
            "oracle25_small_ap_gain_retention": retention,
            "oracle25_all_ap_relative_percent": relative_change(o_all, b_all),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--high-res", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in {
        "baseline": args.baseline,
        "high_res": args.high_res,
        "oracle": args.oracle,
    }.items()}
    report = evaluate_gate(inputs["baseline"], inputs["high_res"], inputs["oracle"])
    report["status"] = "continue" if report["passed"] else "stop"
    report["inputs"] = {
        name: {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for name, path in {"baseline": args.baseline, "high_res": args.high_res, "oracle": args.oracle}.items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
