"""Estimate the fixed-budget oracle ceiling for adaptive 640/960 routing."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from ultralytics.data.utils import check_det_dataset

from scripts.evaluate_object_sizes import AREA_RANGES, IOU_THRESHOLDS, ImageDetections, evaluate, image_matches, load_predictions
from scripts.run_sparse_refinement_feasibility import BDD_NAMES


def small_match_utility(baseline: ImageDetections, high_res: ImageDetections, max_det: int = 300) -> int:
    """Score extra class-aware small-GT matches across the COCO IoU thresholds."""
    utility = 0
    classes = np.union1d(baseline.gt_classes, high_res.gt_classes)
    for class_id in classes:
        for threshold in IOU_THRESHOLDS:
            _, baseline_tp, _ = image_matches(baseline, int(class_id), AREA_RANGES["small"], threshold, max_det)
            _, high_res_tp, _ = image_matches(high_res, int(class_id), AREA_RANGES["small"], threshold, max_det)
            utility += int(np.sum(high_res_tp)) - int(np.sum(baseline_tp))
    return utility


def select_oracle_routes(
    baseline: list[ImageDetections],
    high_res: list[ImageDetections],
    fraction: float,
    max_det: int = 300,
) -> tuple[set[int], list[int]]:
    """Select exactly the fixed-budget fraction with stable utility tie-breaking."""
    if len(baseline) != len(high_res):
        raise ValueError("prediction sets must contain the same number of images")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be in [0, 1]")
    utilities = [small_match_utility(first, second, max_det) for first, second in zip(baseline, high_res)]
    return select_from_utilities(utilities, fraction), utilities


def select_from_utilities(utilities: list[int], fraction: float) -> set[int]:
    """Return stable top-utility indices for a fixed routing fraction."""
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be in [0, 1]")
    count = math.ceil(fraction * len(utilities))
    order = sorted(range(len(utilities)), key=lambda index: (-utilities[index], index))
    return set(order[:count])


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--high-res", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, help="Dataset YAML used to resolve class names")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.25, 0.5])
    parser.add_argument("--labels-output", type=Path)
    parser.add_argument("--max-det", type=int, default=300)
    args = parser.parse_args()

    names = BDD_NAMES
    if args.data:
        data = check_det_dataset(str(args.data.resolve()), autodownload=False, split=args.split)
        raw_names = data["names"]
        names = (
            {int(index): str(name) for index, name in raw_names.items()}
            if isinstance(raw_names, dict)
            else dict(enumerate(raw_names))
        )

    baseline_records = load_predictions(args.baseline)
    high_res_records = load_predictions(args.high_res)
    baseline_lines = read_jsonl(args.baseline)
    high_res_lines = read_jsonl(args.high_res)
    baseline_images = [item["image"] for item in baseline_lines]
    if baseline_images != [item["image"] for item in high_res_lines]:
        raise ValueError("prediction JSONL image order differs")

    utilities = [
        small_match_utility(first, second, args.max_det)
        for first, second in zip(baseline_records, high_res_records)
    ]
    order = sorted(range(len(utilities)), key=lambda index: (-utilities[index], index))
    ranks = {index: rank for rank, index in enumerate(order, 1)}
    labels_output = args.labels_output or args.output.with_suffix(".labels.jsonl")
    labels_output.parent.mkdir(parents=True, exist_ok=True)
    with labels_output.open("w", encoding="utf-8") as output_file:
        for index, image in enumerate(baseline_images):
            output_file.write(
                json.dumps(
                    {
                        "image": image,
                        "utility": utilities[index],
                        "rank": ranks[index],
                        "rank_fraction": ranks[index] / len(utilities),
                    }
                )
                + "\n"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "GT oracle development ceiling; not an inference method",
        "baseline": str(args.baseline.resolve()),
        "high_res": str(args.high_res.resolve()),
        "baseline_sha256": hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
        "high_res_sha256": hashlib.sha256(args.high_res.read_bytes()).hexdigest(),
        "images": len(baseline_records),
        "data": str(args.data.resolve()) if args.data else None,
        "split": args.split,
        "names": names,
        "routing_utility": "extra class-aware small-GT matches over IoU 0.50:0.95",
        "labels": str(labels_output.resolve()),
        "labels_sha256": hashlib.sha256(labels_output.read_bytes()).hexdigest(),
        "utility_distribution": {
            "positive": int(np.sum(np.asarray(utilities) > 0)),
            "zero": int(np.sum(np.asarray(utilities) == 0)),
            "negative": int(np.sum(np.asarray(utilities) < 0)),
            "minimum": min(utilities),
            "maximum": max(utilities),
        },
        "budgets": {},
    }
    for fraction in args.fractions:
        selected = select_from_utilities(utilities, fraction)
        key = f"{fraction:.2f}"
        prediction_path = args.output.with_name(f"{args.output.stem}-route-{int(round(100 * fraction)):02d}.predictions.jsonl")
        with prediction_path.open("w", encoding="utf-8") as output_file:
            for index, baseline_line in enumerate(baseline_lines):
                output_file.write(json.dumps(high_res_lines[index] if index in selected else baseline_line) + "\n")
        mixed_records = load_predictions(prediction_path)
        report["budgets"][key] = {
            "route_fraction": fraction,
            "routed_images": len(selected),
            "positive_utility_images": int(np.sum(np.asarray(utilities) > 0)),
            "zero_utility_images": int(np.sum(np.asarray(utilities) == 0)),
            "negative_utility_images": int(np.sum(np.asarray(utilities) < 0)),
            "selected_utility_sum": int(sum(utilities[index] for index in selected)),
            "raw_predictions": str(prediction_path.resolve()),
            "metrics": evaluate(mixed_records, names, args.max_det, progress=True),
        }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
