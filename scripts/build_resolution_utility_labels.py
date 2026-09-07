"""Build per-image marginal-resolution utility labels with bounded memory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.analyze_resolution_routing_oracle import small_match_utility
from scripts.evaluate_object_sizes import ImageDetections, load_ground_truth


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prediction_record(item: dict) -> ImageDetections:
    image = Path(item["image"])
    width, height = int(item["width"]), int(item["height"])
    gt_boxes, gt_classes, gt_areas = load_ground_truth(image, width, height)
    pred_boxes = np.asarray(item["boxes_xyxy"], dtype=np.float32).reshape(-1, 4)
    pred_classes = np.asarray(item["classes"], dtype=np.int64)
    pred_scores = np.asarray(item["scores"], dtype=np.float32)
    pred_areas = np.clip(pred_boxes[:, 2:] - pred_boxes[:, :2], 0.0, None).prod(axis=1)
    return ImageDetections(gt_boxes, gt_classes, gt_areas, pred_boxes, pred_classes, pred_scores, pred_areas)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--high-res", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-det", type=int, default=300)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite labels: {args.output}")
    images, utilities = [], []
    with args.baseline.open(encoding="utf-8") as baseline_file, args.high_res.open(encoding="utf-8") as high_file:
        for index, (baseline_line, high_line) in enumerate(zip(baseline_file, high_file, strict=True), 1):
            baseline_item, high_item = json.loads(baseline_line), json.loads(high_line)
            if baseline_item["image"] != high_item["image"]:
                raise ValueError(f"prediction image order differs at line {index}")
            images.append(baseline_item["image"])
            utilities.append(small_match_utility(prediction_record(baseline_item), prediction_record(high_item), args.max_det))
            if index % 500 == 0:
                print(f"utilities: {index}", flush=True)
    order = sorted(range(len(utilities)), key=lambda index: (-utilities[index], index))
    ranks = {index: rank for rank, index in enumerate(order, 1)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output_file:
        for index, (image, utility) in enumerate(zip(images, utilities)):
            output_file.write(json.dumps({
                "image": image,
                "utility": utility,
                "rank": ranks[index],
                "rank_fraction": ranks[index] / len(utilities),
            }) + "\n")
    values = np.asarray(utilities)
    metadata = {
        "status": "train-only marginal-resolution labels; not validation evidence",
        "images": len(images),
        "baseline_sha256": sha256(args.baseline),
        "high_res_sha256": sha256(args.high_res),
        "labels_sha256": sha256(args.output),
        "utility_distribution": {
            "positive": int(np.sum(values > 0)),
            "zero": int(np.sum(values == 0)),
            "negative": int(np.sum(values < 0)),
            "minimum": int(values.min()),
            "maximum": int(values.max()),
            "mean": float(values.mean()),
        },
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
