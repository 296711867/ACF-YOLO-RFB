"""Evaluate frozen predictions on BDD100K timeofday attribute slices.

Reads completed per-image prediction JSONL files produced by
``scripts/evaluate_object_sizes.py`` and recomputes the identical COCO-style
size metrics (same matching code, same 101-point AP, same area ranges) on
timeofday attribute subsets (daytime / night / dawn-dusk). No inference is
performed and no frozen evidence file is modified; every input JSONL keeps
its original metrics untouched.

Attribute provenance: the official ``bdd100k_labels_images_val.json``
scalabel file carries authoritative ``attributes.timeofday`` per image.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from .evaluate_object_sizes import (
        AREA_RANGES,
        ImageDetections,
        evaluate,
        load_ground_truth,
    )
except ImportError:  # Direct script execution.
    from evaluate_object_sizes import (
        AREA_RANGES,
        ImageDetections,
        evaluate,
        load_ground_truth,
    )

SLICES = ("daytime", "night", "dawn/dusk", "undefined")


def load_timeofday(attributes_source: Path) -> dict[str, str]:
    """Return {image_name.jpg: timeofday} from official scalabel val labels.

    Accepts either the per-image scalabel JSON directory (current official
    ``bdd100k_labels.zip`` layout: ``100k/val/*.json`` with image-level
    ``attributes``) or a single legacy JSON array file.
    """
    table: dict[str, str] = {}
    if attributes_source.is_dir():
        files = sorted(attributes_source.glob("*.json"))
        if not files:
            raise FileNotFoundError(f"No scalabel JSON files under {attributes_source}")
        for path in files:
            entry = json.loads(path.read_text(encoding="utf-8"))
            attributes = entry.get("attributes") or {}
            table[entry["name"] + ".jpg"] = attributes.get("timeofday") or "undefined"
        return table
    entries = json.loads(attributes_source.read_text(encoding="utf-8"))
    for entry in entries:
        attributes = entry.get("attributes") or {}
        table[entry["name"]] = attributes.get("timeofday") or "undefined"
    return table


def load_records_with_names(predictions_path: Path) -> list[tuple[str, ImageDetections]]:
    """Rebuild (image name, record) pairs from one prediction JSONL file."""
    records = []
    with predictions_path.open(encoding="utf-8") as file:
        for line in file:
            prediction = json.loads(line)
            image_path = Path(prediction["image"])
            width, height = int(prediction["width"]), int(prediction["height"])
            gt_boxes, gt_classes, gt_areas = load_ground_truth(image_path, width, height)
            pred_boxes = np.asarray(prediction["boxes_xyxy"], dtype=np.float32).reshape(-1, 4)
            pred_classes = np.asarray(prediction["classes"], dtype=np.int64)
            pred_scores = np.asarray(prediction["scores"], dtype=np.float32)
            pred_areas = np.clip(pred_boxes[:, 2:] - pred_boxes[:, :2], 0.0, None).prod(axis=1)
            records.append(
                (
                    image_path.name,
                    ImageDetections(
                        gt_boxes=gt_boxes,
                        gt_classes=gt_classes,
                        gt_areas=gt_areas,
                        pred_boxes=pred_boxes,
                        pred_classes=pred_classes,
                        pred_scores=pred_scores,
                        pred_areas=pred_areas,
                    ),
                )
            )
    return records


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--attributes", type=Path, required=True, help="Official scalabel val JSON with attributes.timeofday")
    parser.add_argument("--predictions", type=Path, required=True, action="append", help="Frozen predictions JSONL (repeatable)")
    parser.add_argument("--names", type=str, default="person,rider,car,truck,bus,train,motor,bike,traffic light,traffic sign")
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Compute size metrics per timeofday slice for each frozen prediction set."""
    args = parse_args()
    timeofday = load_timeofday(args.attributes)
    names = dict(enumerate([name.strip() for name in args.names.split(",")]))
    report: dict = {
        "protocol": "COCO-style 101-point AP, IoU 0.50:0.95, maxDet 300; identical matching code to evaluate_object_sizes.py",
        "area_ranges_pixels": {
            key: [value[0], None if value[1] == float("inf") else value[1]] for key, value in AREA_RANGES.items()
        },
        "attributes_source": str(args.attributes.resolve()),
        "slice_image_counts": {},
        "models": {},
    }
    for predictions_path in args.predictions:
        records = load_records_with_names(predictions_path.resolve())
        by_slice: dict[str, list[ImageDetections]] = {slice_name: [] for slice_name in SLICES}
        unmatched = []
        for image_name, record in records:
            slice_name = timeofday.get(image_name)
            if slice_name is None:
                unmatched.append(image_name)
                continue
            by_slice[slice_name].append(record)
        if unmatched:
            raise RuntimeError(
                f"{len(unmatched)} prediction images missing from attribute table "
                f"(first few: {unmatched[:5]}): {predictions_path}"
            )
        model_key = predictions_path.stem.replace(".predictions", "")
        report["slice_image_counts"] = {slice_name: len(by_slice[slice_name]) for slice_name in SLICES}
        report["models"][model_key] = {
            "raw_predictions": str(predictions_path.resolve()),
            "slices": {
                slice_name: evaluate(by_slice[slice_name], names, args.max_det)
                for slice_name in SLICES
                if by_slice[slice_name]
            },
        }
        print(f"processed {predictions_path.name}: " + json.dumps(report["slice_image_counts"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "models": list(report["models"])}, indent=2))


if __name__ == "__main__":
    main()
