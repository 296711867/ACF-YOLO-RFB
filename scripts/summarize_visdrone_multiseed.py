"""Aggregate frozen VisDrone validation evidence across seeds 0, 1, and 2."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEEDS = {
    0: {
        "run": ROOT / "runs/detect/phase3-visdrone/visdrone-yolo26n-640-e100-stop20-b16-seed0",
        "size": ROOT / "results/visdrone-native-e100/yolo26n-e100-size-val.json",
        "official": ROOT / "results/visdrone-native-e100/yolo26n-e100-official-compatible-val.json",
    },
    1: {
        "run": ROOT / "runs/detect/phase3-visdrone/visdrone-yolo26n-640-e100-b16-seed1",
        "size": ROOT / "results/visdrone-multiseed/seed1-size-val.json",
        "official": ROOT / "results/visdrone-multiseed/seed1-official-compatible-val.json",
    },
    2: {
        "run": ROOT / "runs/detect/phase3-visdrone/visdrone-yolo26n-640-e100-b16-seed2",
        "size": ROOT / "results/visdrone-multiseed/seed2-size-val.json",
        "official": ROOT / "results/visdrone-multiseed/seed2-official-compatible-val.json",
    },
}


def sha256(path: Path) -> str:
    """Return a file SHA256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(values: list[float]) -> dict:
    """Return reproducible descriptive statistics with sample standard deviation."""
    mean = statistics.mean(values)
    standard_deviation = statistics.stdev(values)
    return {
        "values": values,
        "mean": mean,
        "sample_std": standard_deviation,
        "min": min(values),
        "max": max(values),
        "coefficient_of_variation_percent": standard_deviation / mean * 100 if mean else None,
    }


def cumulative_training_seconds(rows: list[dict]) -> float:
    """Sum cumulative timer segments, including timers reset by a formal resume."""
    elapsed = [float(row["time"]) for row in rows]
    return sum(previous for previous, current in zip(elapsed, elapsed[1:]) if current < previous) + elapsed[-1]


def load_seed(seed: int, paths: dict[str, Path]) -> dict:
    """Load one seed's training, size, official, and checkpoint evidence."""
    run_dir = paths["run"]
    rows = list(csv.DictReader((run_dir / "results.csv").open(encoding="utf-8")))
    if len(rows) != 100 or int(rows[-1]["epoch"]) != 100:
        raise RuntimeError(f"Seed {seed} training is incomplete: rows={len(rows)}")
    metric_key = "metrics/mAP50-95(B)"
    best = max(rows, key=lambda row: float(row[metric_key]))
    metadata = json.loads((run_dir / "experiment_metadata.json").read_text(encoding="utf-8"))
    size = json.loads(paths["size"].read_text(encoding="utf-8"))
    official = json.loads(paths["official"].read_text(encoding="utf-8"))
    return {
        "seed": seed,
        "run": str(run_dir),
        "completed_epochs": len(rows),
        "training_seconds": cumulative_training_seconds(rows),
        "best_epoch": int(best["epoch"]),
        "training_best": {
            "precision": float(best["metrics/precision(B)"]),
            "recall": float(best["metrics/recall(B)"]),
            "mAP50": float(best["metrics/mAP50(B)"]),
            "mAP50_95": float(best[metric_key]),
        },
        "best_weight_sha256": metadata["outputs"]["best"]["sha256"],
        "size_metrics": size["metrics"],
        "official_metrics": official["metrics"],
        "evidence_sha256": {
            "results_csv": sha256(run_dir / "results.csv"),
            "size_json": sha256(paths["size"]),
            "official_json": sha256(paths["official"]),
        },
    }


def aggregate(records: list[dict]) -> dict:
    """Aggregate training, size, and official metrics across seed records."""
    training = {
        key: summarize([record["training_best"][key] for record in records])
        for key in ("precision", "recall", "mAP50", "mAP50_95")
    }
    training["seconds"] = summarize([record["training_seconds"] for record in records])
    size = {
        area: {
            key: summarize([record["size_metrics"][area][key] for record in records])
            for key in ("AP50_95", "AP50", "AR50_95", "AR50")
        }
        for area in ("all", "small", "medium", "large")
    }
    official = {
        key: summarize([record["official_metrics"][key] for record in records])
        for key in ("AP50_95", "AP50", "AP75", "AR1", "AR10", "AR100", "AR500")
    }
    return {"training_best": training, "size": size, "official_compatible_percent": official}


def main() -> None:
    """Build and save the frozen multi-seed report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/visdrone-multiseed/summary.json"))
    args = parser.parse_args()
    records = [load_seed(seed, paths) for seed, paths in SEEDS.items()]
    report = {
        "protocol": "VisDrone native YOLO26n, seeds 0/1/2, frozen train/val only; test-dev excluded",
        "statistics": "arithmetic mean and sample standard deviation (n-1)",
        "seeds": records,
        "aggregate": aggregate(records),
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["aggregate"], indent=2))


if __name__ == "__main__":
    main()
