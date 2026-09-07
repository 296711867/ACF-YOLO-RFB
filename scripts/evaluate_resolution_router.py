"""Evaluate learned and non-learned fixed-budget 640/960 routing policies."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.evaluate_object_sizes import evaluate, load_predictions
from scripts.run_sparse_refinement_feasibility import BDD_NAMES
from scripts.train_perspective_router import top_fraction_indices


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def learned_scores(path: Path, images: list[str]) -> np.ndarray:
    """Align filename-keyed router scores to prediction order."""
    values = {Path(item["image"]).name: float(item["score"]) for item in read_jsonl(path)}
    missing = [image for image in images if Path(image).name not in values]
    if missing:
        raise ValueError(f"learned scores missing image: {missing[0]}")
    return np.asarray([values[Path(image).name] for image in images], dtype=np.float64)


def hash_random_scores(images: list[str], seed: int = 0) -> np.ndarray:
    """Return deterministic pseudo-random scores without reading labels."""
    return np.asarray(
        [int.from_bytes(hashlib.sha256(f"{seed}:{Path(image).name}".encode()).digest()[:8], "big") for image in images],
        dtype=np.float64,
    )


def coarse_small_scores(predictions: list[dict], max_area: float = float(32**2)) -> np.ndarray:
    """Sum confidence of coarse predictions already small in original pixels."""
    values = []
    for prediction in predictions:
        boxes = np.asarray(prediction["boxes_xyxy"], dtype=np.float32).reshape(-1, 4)
        scores = np.asarray(prediction["scores"], dtype=np.float32)
        areas = np.clip(boxes[:, 2:] - boxes[:, :2], 0.0, None).prod(axis=1)
        values.append(float(scores[areas < max_area].sum()))
    return np.asarray(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--high-res", type=Path, required=True)
    parser.add_argument("--learned-scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--route-fraction", type=float, default=0.25)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--status", default="fixed development comparison; not final validation")
    args = parser.parse_args()

    baseline = read_jsonl(args.baseline)
    high_res = read_jsonl(args.high_res)
    images = [item["image"] for item in baseline]
    if images != [item["image"] for item in high_res]:
        raise ValueError("prediction JSONL image order differs")
    policies = {
        "learned_p2_bands": learned_scores(args.learned_scores, images),
        "coarse_small_score_sum": coarse_small_scores(baseline),
        "hash_random_seed0": hash_random_scores(images, args.random_seed),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "status": args.status,
        "route_fraction": args.route_fraction,
        "images": len(images),
        "baseline_sha256": hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
        "high_res_sha256": hashlib.sha256(args.high_res.read_bytes()).hexdigest(),
        "learned_scores_sha256": hashlib.sha256(args.learned_scores.read_bytes()).hexdigest(),
        "policies": {},
    }
    for name, scores in policies.items():
        selected = set(top_fraction_indices(scores, args.route_fraction).tolist())
        prediction_path = args.output.with_name(f"{args.output.stem}-{name}.predictions.jsonl")
        with prediction_path.open("w", encoding="utf-8") as output_file:
            for index, item in enumerate(baseline):
                output_file.write(json.dumps(high_res[index] if index in selected else item) + "\n")
        report["policies"][name] = {
            "routed_images": len(selected),
            "raw_predictions": str(prediction_path.resolve()),
            "metrics": evaluate(load_predictions(prediction_path), BDD_NAMES, args.max_det, progress=True),
        }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
