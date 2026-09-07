"""Python port of the official VisDrone DET toolkit v1.0.4 evaluation logic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


IOU_THRESHOLDS = np.arange(0.5, 0.96, 0.05)
MAX_DETECTIONS = (1, 10, 100, 500)
TOOLKIT_URL = "https://github.com/VisDrone/VisDrone2018-DET-toolkit"
TOOLKIT_ZIP_SHA256 = "30298f4e0b56dcb5a49ab4a0d790b2c3329da5fa4391757920b74294351ff3b6"


def load_rows(path: Path) -> np.ndarray:
    """Load a comma-delimited VisDrone file while accepting an empty detection file."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return np.zeros((0, 8), dtype=np.float64)
    rows = np.asarray([[float(value) for value in line.split(",")] for line in text.splitlines()], dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] < 6 or not np.isfinite(rows).all():
        raise ValueError(f"Invalid VisDrone rows in {path}")
    return rows


def matlab_round_positive(values: np.ndarray) -> np.ndarray:
    """Match MATLAB round for the non-negative coordinates used by VisDrone."""
    return np.floor(values + 0.5).astype(np.int64)


def integral_image(mask: np.ndarray) -> np.ndarray:
    """Match createIntImg.m for a two-dimensional mask."""
    return mask.astype(np.float64).cumsum(axis=0).cumsum(axis=1)


def ignore_fraction(box: np.ndarray, integral: np.ndarray, width: int, height: int) -> float:
    """Match dropObjectsInIgr.m, including its integral-image coordinate convention."""
    x, y, box_width, box_height = np.maximum(1, matlab_round_positive(np.asarray(box, dtype=np.float64)))
    x = max(1, min(width, int(x)))
    y = max(1, min(height, int(y)))
    right = min(width, x + int(box_width))
    bottom = min(height, y + int(box_height))

    def value(row: int, column: int) -> float:
        return float(integral[row - 1, column - 1])

    ignored = value(y, x) + value(bottom, right) - value(y, right) - value(bottom, x)
    return ignored / max(1.0, float(box_width * box_height))


def drop_objects_in_ignored_regions(
    ground_truth: np.ndarray, detections: np.ndarray, image_height: int, image_width: int
) -> tuple[np.ndarray, np.ndarray]:
    """Port official dropObjectsInIgr.m."""
    current_gt = ground_truth[ground_truth[:, 5] != 0]
    ignored_regions = np.maximum(1, ground_truth[ground_truth[:, 5] == 0, :4])
    if not len(ignored_regions):
        return current_gt, detections

    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    for left, top, width, height in ignored_regions.astype(np.int64):
        row_start = max(1, top) - 1
        row_end = min(image_height, top + height)
        col_start = max(1, left) - 1
        col_end = min(image_width, left + width)
        mask[row_start:row_end, col_start:col_end] = 1
    integral = integral_image(mask)
    current_gt = np.asarray(
        [row for row in current_gt if ignore_fraction(row[:4], integral, image_width, image_height) < 0.5],
        dtype=np.float64,
    ).reshape(-1, ground_truth.shape[1])
    detections = np.asarray(
        [row for row in detections if ignore_fraction(row[:4], integral, image_width, image_height) < 0.5],
        dtype=np.float64,
    ).reshape(-1, detections.shape[1])
    return current_gt, detections


def overlap_matrix(detections: np.ndarray, ground_truth: np.ndarray, ignored: np.ndarray) -> np.ndarray:
    """Port official compOas.m modified Pascal overlap."""
    if not len(detections) or not len(ground_truth):
        return np.zeros((len(detections), len(ground_truth)), dtype=np.float64)
    det_end = detections[:, None, :2] + detections[:, None, 2:4]
    gt_end = ground_truth[None, :, :2] + ground_truth[None, :, 2:4]
    extent = np.minimum(det_end, gt_end) - np.maximum(detections[:, None, :2], ground_truth[None, :, :2])
    intersection = np.maximum(extent[..., 0], 0) * np.maximum(extent[..., 1], 0)
    det_area = (detections[:, 2] * detections[:, 3])[:, None]
    gt_area = (ground_truth[:, 2] * ground_truth[:, 3])[None, :]
    union = np.where(ignored[None, :], det_area, det_area + gt_area - intersection)
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def evaluate_image(ground_truth: np.ndarray, detections: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Port official evalRes.m for one class in one image."""
    ground_truth = np.asarray(ground_truth, dtype=np.float64).reshape(-1, 5)
    detections = np.asarray(detections, dtype=np.float64).reshape(-1, 5)
    detections = detections[np.argsort(-detections[:, 4], kind="stable")] if len(detections) else detections
    ground_truth = ground_truth[np.argsort(ground_truth[:, 4], kind="stable")] if len(ground_truth) else ground_truth
    gt = ground_truth.copy()
    gt[:, 4] = -gt[:, 4]
    det = np.column_stack((detections, np.zeros(len(detections), dtype=np.float64)))
    overlaps = overlap_matrix(det[:, :4], gt[:, :4], gt[:, 4] == -1)

    for det_index in range(len(det)):
        best_overlap = threshold
        best_gt = -1
        best_match = 0
        for gt_index in range(len(gt)):
            match = gt[gt_index, 4]
            if match == 1:
                continue
            if best_match != 0 and match == -1:
                break
            if overlaps[det_index, gt_index] < best_overlap:
                continue
            best_overlap = overlaps[det_index, gt_index]
            best_gt = gt_index
            best_match = 1 if match == 0 else -1
        if best_match == -1:
            det[det_index, 5] = -1
        elif best_match == 1:
            gt[best_gt, 4] = 1
            det[det_index, 5] = 1
    return gt, det


def voc_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """Port official VOCap.m continuous interpolation."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for index in range(len(mpre) - 2, -1, -1):
        mpre[index] = max(mpre[index], mpre[index + 1])
    changing = np.flatnonzero(mrec[1:] != mrec[:-1]) + 1
    return float(np.sum((mrec[changing] - mrec[changing - 1]) * mpre[changing]))


def calculate_accuracy(all_gt: list[np.ndarray], all_det: list[np.ndarray]) -> dict:
    """Port official calcAccuracy.m and expose aggregate and per-class metrics."""
    ap = np.zeros((10, len(IOU_THRESHOLDS)), dtype=np.float64)
    ar = np.zeros((10, len(IOU_THRESHOLDS), len(MAX_DETECTIONS)), dtype=np.float64)
    class_weights = np.asarray(
        [sum(bool(np.count_nonzero(gt[:, 5] == class_id)) for gt in all_gt) for class_id in range(1, 11)],
        dtype=np.float64,
    )

    for class_index, class_id in enumerate(range(1, 11)):
        for threshold_index, threshold in enumerate(IOU_THRESHOLDS):
            final_recall = np.zeros(0, dtype=np.float64)
            final_precision = np.zeros(0, dtype=np.float64)
            for max_det_index, max_det in enumerate(MAX_DETECTIONS):
                gt_matches = []
                det_matches = []
                for gt, det in zip(all_gt, all_det):
                    gt_for_class = gt[gt[:, 5] == class_id, :5]
                    top_detections = det[: min(len(det), max_det)]
                    det_for_class = top_detections[top_detections[:, 5] == class_id, :5]
                    matched_gt, matched_det = evaluate_image(gt_for_class, det_for_class, float(threshold))
                    gt_matches.append(matched_gt[:, 4])
                    det_matches.append(matched_det[:, 4:6])
                gt_match = np.concatenate(gt_matches) if gt_matches else np.zeros(0)
                det_match = np.concatenate(det_matches) if det_matches else np.zeros((0, 2))
                order = np.argsort(-det_match[:, 0], kind="stable") if len(det_match) else np.zeros(0, dtype=int)
                true_positives = np.cumsum(det_match[order, 1] == 1)
                recall = true_positives / max(1, len(gt_match))
                if len(recall):
                    ar[class_index, threshold_index, max_det_index] = float(np.max(recall)) * 100
                false_positives = np.cumsum(det_match[order, 1] == 0)
                precision = true_positives / np.maximum(1, false_positives + true_positives)
                if max_det == 500:
                    final_recall, final_precision = recall, precision
            ap[class_index, threshold_index] = voc_ap(final_recall, final_precision) * 100

    available = class_weights > 0
    weights = class_weights[available]

    def weighted_mean(values: np.ndarray) -> float:
        expanded = weights.reshape((-1,) + (1,) * (values.ndim - 1))
        return float(np.sum(values[available] * expanded) / (np.sum(weights) * np.prod(values.shape[1:])))

    per_class = {
        str(class_id): {
            "images_with_ground_truth": int(class_weights[class_id - 1]),
            "AP50_95": float(np.mean(ap[class_id - 1])),
            "AP50": float(ap[class_id - 1, 0]),
            "AP75": float(ap[class_id - 1, 5]),
            "AR1": float(np.mean(ar[class_id - 1, :, 0])),
            "AR10": float(np.mean(ar[class_id - 1, :, 1])),
            "AR100": float(np.mean(ar[class_id - 1, :, 2])),
            "AR500": float(np.mean(ar[class_id - 1, :, 3])),
        }
        for class_id in range(1, 11)
    }
    return {
        "unit": "percent",
        "AP50_95": weighted_mean(ap),
        "AP50": weighted_mean(ap[:, :1]),
        "AP75": weighted_mean(ap[:, 5:6]),
        "AR1": weighted_mean(ar[:, :, :1]),
        "AR10": weighted_mean(ar[:, :, 1:2]),
        "AR100": weighted_mean(ar[:, :, 2:3]),
        "AR500": weighted_mean(ar[:, :, 3:4]),
        "per_class": per_class,
    }


def evaluate(annotation_dir: Path, prediction_dir: Path, image_dir: Path) -> dict:
    """Evaluate a complete VisDrone split with official v1.0.4-compatible logic."""
    annotation_paths = sorted(annotation_dir.resolve().glob("*.txt"))
    if not annotation_paths:
        raise FileNotFoundError(f"No annotations found in {annotation_dir}")
    expected = {path.name for path in annotation_paths}
    actual = {path.name for path in prediction_dir.resolve().glob("*.txt")}
    if expected != actual:
        raise RuntimeError(
            f"Prediction files must exactly match annotations: missing={sorted(expected - actual)[:5]}, "
            f"extra={sorted(actual - expected)[:5]}"
        )

    all_gt, all_det = [], []
    for annotation_path in annotation_paths:
        detections = load_rows(prediction_dir / annotation_path.name)
        if len(detections) and np.any(detections[:-1, 4] < detections[1:, 4]):
            raise ValueError(f"Detections must be globally score-sorted for official maxDets behavior: {annotation_path.name}")
        ground_truth = load_rows(annotation_path)
        image_path = image_dir / f"{annotation_path.stem}.jpg"
        with Image.open(image_path) as image:
            width, height = image.size
        ground_truth, detections = drop_objects_in_ignored_regions(ground_truth, detections, height, width)
        gt = ground_truth.copy()
        if len(gt):
            if np.any(~np.isin(gt[:, 4], (0, 1))):
                raise ValueError(f"Unexpected ground-truth score in {annotation_path}")
            gt[:, 4] = np.where(gt[:, 4] == 0, 1, 0)
        all_gt.append(gt)
        all_det.append(detections)

    metrics = calculate_accuracy(all_gt, all_det)
    return {
        "implementation": "Python port of official VisDrone DET toolkit v1.0.4 MATLAB logic",
        "official_toolkit_url": TOOLKIT_URL,
        "official_toolkit_zip_sha256": TOOLKIT_ZIP_SHA256,
        "images": len(annotation_paths),
        "annotations": str(annotation_dir.resolve()),
        "predictions": str(prediction_dir.resolve()),
        "image_dir": str(image_dir.resolve()),
        "metrics": metrics,
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = evaluate(arguments.annotations, arguments.predictions, arguments.images)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
