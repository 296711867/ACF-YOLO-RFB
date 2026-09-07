"""Paired image-level bootstrap for frozen BDD100K detection predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scripts.evaluate_object_sizes import (
    AREA_RANGES,
    IOU_THRESHOLDS,
    ImageDetections,
    image_matches,
    interpolated_ap,
    load_ground_truth,
)


ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = ("person", "rider", "car", "truck", "bus", "train", "motor", "bike", "traffic light", "traffic sign")
AREAS = ("all", "small")
_BOOTSTRAP_STATE = None


@dataclass
class RankedDetections:
    """Detection outcomes sorted once by confidence for weighted bootstrap reuse."""

    image_indices: np.ndarray
    true_positive: np.ndarray


@dataclass
class PreparedModel:
    """Per-image positives and ranked detections for one frozen model."""

    positives: dict[tuple[str, int], np.ndarray]
    detections: dict[tuple[str, int, int], RankedDetections]


def sha256(path: Path) -> str:
    """Return a file SHA256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_prediction_records(path: Path) -> tuple[list[str], list[ImageDetections]]:
    """Load raw JSONL predictions and their paired YOLO ground truth."""
    image_keys, records = [], []
    with path.open(encoding="utf-8") as file:
        for line in file:
            item = json.loads(line)
            image_path = Path(item["image"])
            width, height = int(item["width"]), int(item["height"])
            gt_boxes, gt_classes, gt_areas = load_ground_truth(image_path, width, height)
            pred_boxes = np.asarray(item["boxes_xyxy"], dtype=np.float32).reshape(-1, 4)
            pred_classes = np.asarray(item["classes"], dtype=np.int64)
            pred_scores = np.asarray(item["scores"], dtype=np.float32)
            pred_areas = np.clip(pred_boxes[:, 2:] - pred_boxes[:, :2], 0.0, None).prod(axis=1)
            image_keys.append(image_path.name)
            records.append(
                ImageDetections(
                    gt_boxes=gt_boxes,
                    gt_classes=gt_classes,
                    gt_areas=gt_areas,
                    pred_boxes=pred_boxes,
                    pred_classes=pred_classes,
                    pred_scores=pred_scores,
                    pred_areas=pred_areas,
                )
            )
    return image_keys, records


def prepare_model(records: list[ImageDetections], max_det: int = 300) -> PreparedModel:
    """Precompute per-image matches so bootstrap replicates only apply image weights."""
    positives = {}
    detections = {}
    image_count = len(records)
    for area_name in AREAS:
        lower, upper = AREA_RANGES[area_name]
        for class_id in range(len(CLASS_NAMES)):
            positives[(area_name, class_id)] = np.asarray(
                [
                    np.sum((record.gt_classes == class_id) & (record.gt_areas >= lower) & (record.gt_areas < upper))
                    for record in records
                ],
                dtype=np.int32,
            )
            for threshold_index, threshold in enumerate(IOU_THRESHOLDS):
                score_parts, tp_parts, image_parts = [], [], []
                for image_index, record in enumerate(records):
                    scores, true_positive, ignored = image_matches(
                        record, class_id, AREA_RANGES[area_name], float(threshold), max_det
                    )
                    valid = ~ignored
                    score_parts.append(scores[valid])
                    tp_parts.append(true_positive[valid])
                    image_parts.append(np.full(int(np.sum(valid)), image_index, dtype=np.int32))
                scores = np.concatenate(score_parts)
                true_positive = np.concatenate(tp_parts)
                image_indices = np.concatenate(image_parts)
                order = np.argsort(-scores, kind="stable")
                detections[(area_name, class_id, threshold_index)] = RankedDetections(
                    image_indices=image_indices[order],
                    true_positive=true_positive[order],
                )
    return PreparedModel(positives=positives, detections=detections)


def weighted_ap(ranked: RankedDetections, image_weights: np.ndarray, positives: float) -> float:
    """Calculate AP after applying paired bootstrap multiplicities to images."""
    if positives <= 0:
        return float("nan")
    weights = image_weights[ranked.image_indices]
    true_positive = np.cumsum(weights * ranked.true_positive)
    false_positive = np.cumsum(weights * ~ranked.true_positive)
    active = weights > 0
    if not np.any(active):
        return 0.0
    recall = true_positive[active] / positives
    precision = true_positive[active] / np.maximum(true_positive[active] + false_positive[active], 1e-12)
    return interpolated_ap(recall, precision)


def model_ap(model: PreparedModel, image_weights: np.ndarray, area_name: str) -> float:
    """Return macro AP50-95 for one weighted image sample."""
    class_values = []
    for class_id in range(len(CLASS_NAMES)):
        original_positives = model.positives[(area_name, class_id)]
        if int(np.sum(original_positives)) == 0:
            continue
        positives = float(image_weights @ original_positives)
        if positives <= 0:
            class_values.append(0.0)
            continue
        threshold_values = [
            weighted_ap(model.detections[(area_name, class_id, index)], image_weights, positives)
            for index in range(len(IOU_THRESHOLDS))
        ]
        class_values.append(float(np.mean(threshold_values)))
    return float(np.mean(class_values))


def bootstrap_one(seed: int) -> dict[str, tuple[float, float, float]]:
    """Evaluate one paired image bootstrap sample using inherited read-only state."""
    if _BOOTSTRAP_STATE is None:
        raise RuntimeError("Bootstrap worker state has not been initialized")
    baseline, candidate, image_count = _BOOTSTRAP_STATE
    rng = np.random.default_rng(seed)
    weights = np.bincount(rng.integers(0, image_count, size=image_count), minlength=image_count).astype(np.float64)
    output = {}
    for area_name in AREAS:
        baseline_ap = model_ap(baseline, weights, area_name)
        candidate_ap = model_ap(candidate, weights, area_name)
        output[area_name] = (baseline_ap, candidate_ap, candidate_ap - baseline_ap)
    return output


def paired_bootstrap(
    baseline: PreparedModel,
    candidate: PreparedModel,
    image_count: int,
    replicates: int,
    seed: int,
    workers: int,
) -> dict:
    """Run paired nonparametric image bootstrap for all and small AP."""
    global _BOOTSTRAP_STATE
    _BOOTSTRAP_STATE = (baseline, candidate, image_count)
    samples = {area: {"baseline": [], "candidate": [], "delta": []} for area in AREAS}
    seeds = np.random.SeedSequence(seed).generate_state(replicates).tolist()
    if workers == 1:
        results = map(bootstrap_one, seeds)
    else:
        context = mp.get_context("fork")
        with context.Pool(processes=workers) as pool:
            results = pool.map(bootstrap_one, seeds)
    for result in results:
        for area_name, values in result.items():
            for key, value in zip(("baseline", "candidate", "delta"), values):
                samples[area_name][key].append(value)
    output = {}
    for area_name, values in samples.items():
        delta = np.asarray(values["delta"])
        output[area_name] = {
            key: {
                "mean": float(np.mean(array)),
                "ci95_percentile": [float(value) for value in np.percentile(array, [2.5, 97.5])],
            }
            for key, array in ((name, np.asarray(items)) for name, items in values.items())
        }
        output[area_name]["probability_delta_gt_zero"] = float(np.mean(delta > 0))
    return output


def parse_args() -> argparse.Namespace:
    """Parse command-line inputs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=500)
    parser.add_argument("--seed", type=int, default=260805)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    """Load frozen predictions, run paired bootstrap, and write an evidence report."""
    args = parse_args()
    print("Loading baseline predictions and ground truth...", flush=True)
    baseline_keys, baseline_records = load_prediction_records(args.baseline.resolve())
    print(f"Loaded {len(baseline_records)} baseline images; loading candidate...", flush=True)
    candidate_keys, candidate_records = load_prediction_records(args.candidate.resolve())
    if baseline_keys != candidate_keys:
        raise SystemExit("Baseline and candidate predictions are not aligned to identical image order.")
    print("Preparing baseline detection matches...", flush=True)
    baseline = prepare_model(baseline_records, args.max_det)
    print("Preparing candidate detection matches...", flush=True)
    candidate = prepare_model(candidate_records, args.max_det)
    print(f"Running {args.replicates} paired bootstrap replicates with {args.workers} workers...", flush=True)
    observed_weights = np.ones(len(baseline_keys), dtype=np.float64)
    observed = {
        area: {
            "baseline_AP50_95": model_ap(baseline, observed_weights, area),
            "candidate_AP50_95": model_ap(candidate, observed_weights, area),
        }
        for area in AREAS
    }
    for values in observed.values():
        values["delta_AP50_95"] = values["candidate_AP50_95"] - values["baseline_AP50_95"]
    report = {
        "protocol": "paired nonparametric image bootstrap; fixed original class set; macro 101-point AP; IoU 0.50:0.95; maxDet 300",
        "images": len(baseline_keys),
        "replicates": args.replicates,
        "seed": args.seed,
        "inputs": {
            "baseline": {"path": str(args.baseline.resolve()), "sha256": sha256(args.baseline.resolve())},
            "candidate": {"path": str(args.candidate.resolve()), "sha256": sha256(args.candidate.resolve())},
        },
        "observed": observed,
        "bootstrap": paired_bootstrap(
            baseline, candidate, len(baseline_keys), args.replicates, args.seed, args.workers
        ),
        "interpretation_limit": (
            "Image bootstrap quantifies validation-sample uncertainty, not training-seed uncertainty. "
            "A class present in the original area stratum but absent from a resample contributes zero AP, preserving the original macro denominator."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "observed": observed, "bootstrap": report["bootstrap"]}, indent=2))


if __name__ == "__main__":
    main()
