"""Deterministic qualitative error analysis for frozen BDD100K predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

from scripts.evaluate_object_sizes import box_iou, load_ground_truth


ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = ("person", "rider", "car", "truck", "bus", "train", "motor", "bike", "traffic light", "traffic sign")
CATEGORIES = ("p2_repaired_small", "shared_small_miss", "p2_added_false_positive")
DISPLAY_NAMES = {
    "p2_repaired_small": "P2 repaired small-object misses",
    "shared_small_miss": "Small objects missed by both models",
    "p2_added_false_positive": "Additional P2 false positives",
}


@dataclass(frozen=True)
class Detections:
    """Boxes, classes, and confidence scores for one image."""

    boxes: np.ndarray
    classes: np.ndarray
    scores: np.ndarray


@dataclass(frozen=True)
class MatchResult:
    """One-to-one detection/ground-truth assignments."""

    pred_to_gt: np.ndarray
    gt_to_pred: np.ndarray


@dataclass
class ImageAnalysis:
    """All data needed to classify and render one aligned prediction pair."""

    image: Path
    width: int
    height: int
    gt: Detections
    baseline: Detections
    p2: Detections
    baseline_matches: MatchResult
    p2_matches: MatchResult


def sha256(path: Path) -> str:
    """Return a file SHA256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _box_key(box: np.ndarray) -> tuple[float, float, float, float]:
    return tuple(float(value) for value in box)


def filter_detections(detections: Detections, confidence: float) -> Detections:
    """Apply the frozen confidence threshold without changing coordinates."""
    keep = detections.scores >= confidence
    return Detections(detections.boxes[keep], detections.classes[keep], detections.scores[keep])


def greedy_match(gt: Detections, predictions: Detections, iou_threshold: float) -> MatchResult:
    """Match same-class boxes one-to-one in deterministic confidence order."""
    pred_to_gt = np.full(len(predictions.boxes), -1, dtype=np.int64)
    gt_to_pred = np.full(len(gt.boxes), -1, dtype=np.int64)
    pred_order = sorted(
        range(len(predictions.boxes)),
        key=lambda index: (-float(predictions.scores[index]), _box_key(predictions.boxes[index]), int(predictions.classes[index])),
    )
    ious = box_iou(predictions.boxes, gt.boxes)
    for pred_index in pred_order:
        eligible = [
            gt_index
            for gt_index in range(len(gt.boxes))
            if gt_to_pred[gt_index] < 0
            and predictions.classes[pred_index] == gt.classes[gt_index]
            and ious[pred_index, gt_index] >= iou_threshold
        ]
        if not eligible:
            continue
        gt_index = min(eligible, key=lambda index: (-float(ious[pred_index, index]), _box_key(gt.boxes[index])))
        pred_to_gt[pred_index] = gt_index
        gt_to_pred[gt_index] = pred_index
    return MatchResult(pred_to_gt=pred_to_gt, gt_to_pred=gt_to_pred)


def classify_events(analysis: ImageAnalysis, iou_threshold: float, small_area: float) -> dict[str, list[dict]]:
    """Classify repaired misses, shared misses, and additional P2 false positives."""
    events = {category: [] for category in CATEGORIES}
    image_name = analysis.image.name
    gt_areas = np.clip(analysis.gt.boxes[:, 2:] - analysis.gt.boxes[:, :2], 0.0, None).prod(axis=1)
    for gt_index in np.flatnonzero(gt_areas < small_area):
        baseline_pred = int(analysis.baseline_matches.gt_to_pred[gt_index])
        p2_pred = int(analysis.p2_matches.gt_to_pred[gt_index])
        common = {
            "image": image_name,
            "class_id": int(analysis.gt.classes[gt_index]),
            "class_name": CLASS_NAMES[int(analysis.gt.classes[gt_index])],
            "gt_index": int(gt_index),
            "gt_box_xyxy": [float(value) for value in analysis.gt.boxes[gt_index]],
            "gt_area_px2": float(gt_areas[gt_index]),
        }
        if baseline_pred < 0 and p2_pred >= 0:
            events["p2_repaired_small"].append(
                common
                | {
                    "p2_prediction_index": p2_pred,
                    "p2_box_xyxy": [float(value) for value in analysis.p2.boxes[p2_pred]],
                    "p2_confidence": float(analysis.p2.scores[p2_pred]),
                }
            )
        elif baseline_pred < 0 and p2_pred < 0:
            events["shared_small_miss"].append(common)

    baseline_fp = np.flatnonzero(analysis.baseline_matches.pred_to_gt < 0)
    p2_fp = np.flatnonzero(analysis.p2_matches.pred_to_gt < 0)
    fp_iou = box_iou(analysis.p2.boxes[p2_fp], analysis.baseline.boxes[baseline_fp])
    for local_index, pred_index in enumerate(p2_fp):
        same_class = analysis.baseline.classes[baseline_fp] == analysis.p2.classes[pred_index]
        corresponding = bool(np.any(same_class & (fp_iou[local_index] >= iou_threshold)))
        if corresponding:
            continue
        class_id = int(analysis.p2.classes[pred_index])
        events["p2_added_false_positive"].append(
            {
                "image": image_name,
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                "p2_prediction_index": int(pred_index),
                "p2_box_xyxy": [float(value) for value in analysis.p2.boxes[pred_index]],
                "p2_confidence": float(analysis.p2.scores[pred_index]),
            }
        )
    return events


def event_sort_key(category: str, event: dict) -> tuple:
    """Return the frozen deterministic ordering for one candidate category."""
    if category == "p2_repaired_small":
        primary = (-event["p2_confidence"],)
        box = event["gt_box_xyxy"]
    elif category == "shared_small_miss":
        primary = (event["gt_area_px2"],)
        box = event["gt_box_xyxy"]
    else:
        primary = (-event["p2_confidence"],)
        box = event["p2_box_xyxy"]
    return primary + (event["image"], *box, event["class_id"])


def select_unique_images(events: list[dict], count: int) -> list[dict]:
    """Select the first event from each new image in already-sorted candidates."""
    selected, seen = [], set()
    for event in events:
        if event["image"] in seen:
            continue
        selected.append(event)
        seen.add(event["image"])
        if len(selected) == count:
            break
    return selected


def load_prediction_pair(
    baseline_path: Path, p2_path: Path, confidence: float, iou_threshold: float
) -> Iterable[ImageAnalysis]:
    """Stream aligned frozen prediction files and load their YOLO ground truth."""
    with baseline_path.open(encoding="utf-8") as baseline_file, p2_path.open(encoding="utf-8") as p2_file:
        for line_number, (baseline_line, p2_line) in enumerate(zip(baseline_file, p2_file, strict=True), 1):
            baseline_item, p2_item = json.loads(baseline_line), json.loads(p2_line)
            if Path(baseline_item["image"]).name != Path(p2_item["image"]).name:
                raise ValueError(f"Prediction alignment differs at line {line_number}")
            if (baseline_item["width"], baseline_item["height"]) != (p2_item["width"], p2_item["height"]):
                raise ValueError(f"Image dimensions differ at line {line_number}")
            image = Path(baseline_item["image"])
            width, height = int(baseline_item["width"]), int(baseline_item["height"])
            gt_boxes, gt_classes, _ = load_ground_truth(image, width, height)
            gt = Detections(gt_boxes, gt_classes, np.ones(len(gt_boxes), dtype=np.float32))

            def predictions(item: dict) -> Detections:
                return filter_detections(
                    Detections(
                        np.asarray(item["boxes_xyxy"], dtype=np.float32).reshape(-1, 4),
                        np.asarray(item["classes"], dtype=np.int64),
                        np.asarray(item["scores"], dtype=np.float32),
                    ),
                    confidence,
                )

            baseline, p2 = predictions(baseline_item), predictions(p2_item)
            yield ImageAnalysis(
                image=image,
                width=width,
                height=height,
                gt=gt,
                baseline=baseline,
                p2=p2,
                baseline_matches=greedy_match(gt, baseline, iou_threshold),
                p2_matches=greedy_match(gt, p2, iou_threshold),
            )


def write_jsonl(path: Path, records: list[dict]) -> None:
    """Write a deterministic machine-readable candidate list."""
    with path.open("w", encoding="utf-8") as file:
        for rank, record in enumerate(records, 1):
            file.write(json.dumps({"rank": rank} | record, sort_keys=True) + "\n")


def crop_bounds(box: list[float], width: int, height: int) -> tuple[float, float, float, float]:
    """Create a stable 16:9 context crop centered on the focal event."""
    x1, y1, x2, y2 = box
    crop_width = min(float(width), max(320.0, (x2 - x1) * 8.0))
    crop_height = max(180.0, (y2 - y1) * 8.0, crop_width * 9.0 / 16.0)
    crop_width = max(crop_width, crop_height * 16.0 / 9.0)
    crop_width, crop_height = min(crop_width, float(width)), min(crop_height, float(height))
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    left = min(max(0.0, center_x - crop_width / 2), width - crop_width)
    top = min(max(0.0, center_y - crop_height / 2), height - crop_height)
    return left, top, left + crop_width, top + crop_height


def _draw_box(ax, box: np.ndarray | list[float], color: str, linewidth: float, linestyle: str = "-") -> None:
    x1, y1, x2, y2 = (float(value) for value in box)
    ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=linewidth, linestyle=linestyle))


def render_category(category: str, selected: list[dict], analyses: dict[str, ImageAnalysis], output_dir: Path) -> list[Path]:
    """Render selected events as GT/baseline/P2 comparison rows."""
    rows = len(selected)
    fig, axes = plt.subplots(rows, 3, figsize=(12.0, 3.2 * rows), squeeze=False)
    for row, event in enumerate(selected):
        analysis = analyses[event["image"]]
        image = np.asarray(Image.open(analysis.image).convert("RGB"))
        focal_box = event.get("gt_box_xyxy", event.get("p2_box_xyxy"))
        bounds = crop_bounds(focal_box, analysis.width, analysis.height)
        for column, (title, detections, matches) in enumerate(
            (("Ground truth", analysis.gt, None), ("YOLO26n", analysis.baseline, analysis.baseline_matches), ("YOLO26n-P2", analysis.p2, analysis.p2_matches))
        ):
            ax = axes[row, column]
            ax.imshow(image)
            for gt_box in analysis.gt.boxes:
                _draw_box(ax, gt_box, "#F2F2F2", 0.7, "--")
            if column == 0:
                for gt_box in analysis.gt.boxes:
                    _draw_box(ax, gt_box, "#2CA02C", 1.0)
            else:
                for pred_index, pred_box in enumerate(detections.boxes):
                    color = "#2CA02C" if matches.pred_to_gt[pred_index] >= 0 else "#D62728"
                    _draw_box(ax, pred_box, color, 1.0)
            if category == "p2_added_false_positive" and column == 2:
                _draw_box(ax, event["p2_box_xyxy"], "#FFD400", 2.5)
            elif "gt_box_xyxy" in event:
                _draw_box(ax, event["gt_box_xyxy"], "#FFD400", 2.5)
            ax.set_xlim(bounds[0], bounds[2])
            ax.set_ylim(bounds[3], bounds[1])
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(title, fontsize=10)
        detail = f"#{row + 1} {event['image']} | {event['class_name']}"
        if "p2_confidence" in event:
            detail += f" | P2 conf {event['p2_confidence']:.3f}"
        else:
            detail += f" | area {event['gt_area_px2']:.1f} px^2"
        axes[row, 0].text(
            0.01,
            0.02,
            detail,
            transform=axes[row, 0].transAxes,
            ha="left",
            va="bottom",
            fontsize=7,
            color="white",
            bbox={"facecolor": "#111111", "edgecolor": "none", "alpha": 0.78, "pad": 2.0},
        )
    fig.suptitle(DISPLAY_NAMES[category] + " (yellow: focal event; green: matched; red: unmatched)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    stem = category.replace("_", "-")
    outputs = [output_dir / f"{stem}.png", output_dir / f"{stem}.pdf"]
    fig.savefig(outputs[0], dpi=300, bbox_inches="tight", metadata={"Software": "BDD frozen-error analysis"})
    fig.savefig(outputs[1], bbox_inches="tight", metadata={"Creator": "BDD frozen-error analysis", "CreationDate": None, "ModDate": None})
    plt.close(fig)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, default=ROOT / "results/baseline/bdd100k-yolo26n-640-size-metrics.predictions.jsonl")
    parser.add_argument("--p2", type=Path, default=ROOT / "results/p2/bdd100k-yolo26n-p2-640-size-metrics.predictions.jsonl")
    parser.add_argument("--bootstrap-summary", type=Path, default=ROOT / "results/bdd-paired-bootstrap/summary.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/bdd-error-analysis")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--small-area", type=float, default=float(32**2))
    parser.add_argument("--selected-per-category", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_path, p2_path = args.baseline.resolve(), args.p2.resolve()
    frozen = json.loads(args.bootstrap_summary.read_text(encoding="utf-8"))["inputs"]
    hashes = {"baseline": sha256(baseline_path), "p2": sha256(p2_path)}
    if hashes["baseline"] != frozen["baseline"]["sha256"] or hashes["p2"] != frozen["candidate"]["sha256"]:
        raise SystemExit("Prediction hashes do not match the frozen paired-bootstrap inputs")

    candidates = {category: [] for category in CATEGORIES}
    selected_analyses = {}
    image_count = 0
    all_analyses = {}
    for analysis in load_prediction_pair(baseline_path, p2_path, args.confidence, args.iou):
        image_count += 1
        events = classify_events(analysis, args.iou, args.small_area)
        if any(events.values()):
            all_analyses[analysis.image.name] = analysis
        for category in CATEGORIES:
            candidates[category].extend(events[category])
    if image_count != 10000:
        raise SystemExit(f"Expected 10000 aligned BDD val images, found {image_count}")

    selected, outputs = {}, []
    for category in CATEGORIES:
        candidates[category].sort(key=lambda event: event_sort_key(category, event))
        candidate_path = output_dir / f"{category}.jsonl"
        write_jsonl(candidate_path, candidates[category])
        outputs.append(candidate_path)
        selected[category] = select_unique_images(candidates[category], args.selected_per_category)
        if len(selected[category]) < args.selected_per_category:
            raise SystemExit(f"Not enough unique images for {category}")
        for event in selected[category]:
            selected_analyses[event["image"]] = all_analyses[event["image"]]
        outputs.extend(render_category(category, selected[category], all_analyses, output_dir))

    selected_inputs = {}
    for image_name, analysis in sorted(selected_analyses.items()):
        label = analysis.image.parents[2] / "labels" / "val" / f"{analysis.image.stem}.txt"
        selected_inputs[image_name] = {
            "image_sha256": sha256(analysis.image),
            "label_sha256": sha256(label),
        }
    manifest = {
        "generator": "scripts/analyze_bdd_errors.py",
        "protocol": {
            "confidence_threshold": args.confidence,
            "iou_threshold": args.iou,
            "small_gt_area_px2_upper_exclusive": args.small_area,
            "matching": "same-class one-to-one greedy matching in descending confidence order",
            "p2_false_positive_correspondence": "same-class unmatched baseline prediction with box IoU >= threshold",
            "selection": {
                "p2_repaired_small": "descending P2 confidence, then image/GT box/class",
                "shared_small_miss": "ascending GT area, then image/GT box/class",
                "p2_added_false_positive": "descending P2 confidence, then image/P2 box/class",
                "images_per_category": args.selected_per_category,
                "deduplication": "first ranked event per previously unseen image",
            },
        },
        "inputs": {
            "images": image_count,
            "baseline_predictions": {"path": str(baseline_path.relative_to(ROOT)), "sha256": hashes["baseline"]},
            "p2_predictions": {"path": str(p2_path.relative_to(ROOT)), "sha256": hashes["p2"]},
            "frozen_hash_source": {"path": str(args.bootstrap_summary.resolve().relative_to(ROOT)), "sha256": sha256(args.bootstrap_summary.resolve())},
            "selected_images_and_labels": selected_inputs,
        },
        "candidate_counts": {
            category: {
                "events": len(candidates[category]),
                "images": len({event["image"] for event in candidates[category]}),
                "per_class": dict(sorted(Counter(event["class_name"] for event in candidates[category]).items())),
            }
            for category in CATEGORIES
        },
        "selected": selected,
        "outputs": {path.name: sha256(path) for path in outputs},
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "candidate_counts": manifest["candidate_counts"], "selected": selected}, indent=2))


if __name__ == "__main__":
    main()
