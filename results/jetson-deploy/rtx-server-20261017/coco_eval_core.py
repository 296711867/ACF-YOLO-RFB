"""COCO-style evaluation core for the frozen-subset consistency check.

The matching logic, 101-point interpolation, area ranges and ignore semantics
are copied verbatim from the research repo's scripts/evaluate_object_sizes.py
(branch p20-lightweight, frozen 2026-09-23). The frozen 10k thesis numbers
(L0 / K1 ep200 size-val JSONs) were produced by that script, so reusing the
identical core here makes the Jetson subset AP *definitionally* comparable.

Dependencies: numpy only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

IOU_THRESHOLDS = np.arange(0.50, 0.96, 0.05)
RECALL_THRESHOLDS = np.linspace(0.0, 1.0, 101)
AREA_RANGES = {
    "all": (0.0, math.inf),
    "small": (0.0, float(32**2)),
    "medium": (float(32**2), float(96**2)),
    "large": (float(96**2), math.inf),
}


@dataclass
class ImageDetections:
    """Ground truth and predictions for one image in pixel xyxy coordinates."""

    gt_boxes: np.ndarray
    gt_classes: np.ndarray
    gt_areas: np.ndarray
    pred_boxes: np.ndarray
    pred_classes: np.ndarray
    pred_scores: np.ndarray
    pred_areas: np.ndarray


def box_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Return pairwise IoU for xyxy arrays."""
    if not len(boxes1) or not len(boxes2):
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)
    top_left = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    bottom_right = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    intersection = np.clip(bottom_right - top_left, 0.0, None).prod(axis=2)
    area1 = np.clip(boxes1[:, 2:] - boxes1[:, :2], 0.0, None).prod(axis=1)
    area2 = np.clip(boxes2[:, 2:] - boxes2[:, :2], 0.0, None).prod(axis=1)
    return intersection / np.maximum(area1[:, None] + area2[None, :] - intersection, 1e-12)


def load_ground_truth_label(label_path: Path, width: int, height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load one normalized YOLO label file and convert to pixel xyxy boxes."""
    boxes, classes = [], []
    if label_path.is_file():
        for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            values = line.split()
            if len(values) != 5:
                raise ValueError(f"{label_path}:{line_number}: expected 5 columns")
            class_id = int(values[0])
            x_center, y_center, box_width, box_height = (float(value) for value in values[1:])
            boxes.append(
                (
                    (x_center - box_width / 2) * width,
                    (y_center - box_height / 2) * height,
                    (x_center + box_width / 2) * width,
                    (y_center + box_height / 2) * height,
                )
            )
            classes.append(class_id)
    boxes_array = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    areas = np.clip(boxes_array[:, 2:] - boxes_array[:, :2], 0.0, None).prod(axis=1)
    return boxes_array, np.asarray(classes, dtype=np.int64), areas


def make_record(
    gt: tuple[np.ndarray, np.ndarray, np.ndarray],
    pred_boxes: np.ndarray,
    pred_classes: np.ndarray,
    pred_scores: np.ndarray,
) -> ImageDetections:
    pred_boxes = np.asarray(pred_boxes, dtype=np.float32).reshape(-1, 4)
    pred_areas = np.clip(pred_boxes[:, 2:] - pred_boxes[:, :2], 0.0, None).prod(axis=1)
    return ImageDetections(
        gt_boxes=gt[0],
        gt_classes=gt[1],
        gt_areas=gt[2],
        pred_boxes=pred_boxes,
        pred_classes=np.asarray(pred_classes, dtype=np.int64),
        pred_scores=np.asarray(pred_scores, dtype=np.float32),
        pred_areas=pred_areas,
    )


def image_matches(
    record: ImageDetections,
    class_id: int,
    area_range: tuple[float, float],
    iou_threshold: float,
    max_det: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match one image using COCO's preference for non-ignored ground truths."""
    pred_indices = np.flatnonzero(record.pred_classes == class_id)
    pred_indices = pred_indices[np.argsort(-record.pred_scores[pred_indices], kind="stable")][:max_det]
    gt_indices = np.flatnonzero(record.gt_classes == class_id)
    lower, upper = area_range
    gt_ignore = (record.gt_areas[gt_indices] < lower) | (record.gt_areas[gt_indices] >= upper)
    order = np.argsort(gt_ignore, kind="stable")
    gt_indices, gt_ignore = gt_indices[order], gt_ignore[order]
    ious = box_iou(record.pred_boxes[pred_indices], record.gt_boxes[gt_indices])
    matched_gt = np.zeros(len(gt_indices), dtype=bool)
    true_positive = np.zeros(len(pred_indices), dtype=bool)
    detection_ignore = np.zeros(len(pred_indices), dtype=bool)

    for detection_index in range(len(pred_indices)):
        best_gt = -1
        best_iou = iou_threshold
        for gt_index in range(len(gt_indices)):
            if matched_gt[gt_index]:
                continue
            if best_gt >= 0 and not gt_ignore[best_gt] and gt_ignore[gt_index]:
                break
            if ious[detection_index, gt_index] < best_iou:
                continue
            best_iou = ious[detection_index, gt_index]
            best_gt = gt_index
        if best_gt >= 0:
            matched_gt[best_gt] = True
            detection_ignore[detection_index] = gt_ignore[best_gt]
            true_positive[detection_index] = not gt_ignore[best_gt]
        else:
            area = record.pred_areas[pred_indices[detection_index]]
            detection_ignore[detection_index] = area < lower or area >= upper
    return record.pred_scores[pred_indices], true_positive, detection_ignore


def interpolated_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """Compute COCO 101-point interpolated average precision."""
    if not len(recall):
        return 0.0
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    values = []
    for threshold in RECALL_THRESHOLDS:
        indices = np.flatnonzero(recall >= threshold)
        values.append(float(precision[indices[0]]) if len(indices) else 0.0)
    return float(np.mean(values))


def evaluate(records: list[ImageDetections], names: dict[int, str], max_det: int, progress: bool = False) -> dict:
    """Compute macro AP and AR for every COCO area range."""
    output = {}
    for area_name, area_range in AREA_RANGES.items():
        if progress:
            print(f"Evaluating {area_name} objects...", flush=True)
        per_class = {}
        for class_id, class_name in names.items():
            lower, upper = area_range
            positives = sum(
                int(np.sum((record.gt_classes == class_id) & (record.gt_areas >= lower) & (record.gt_areas < upper)))
                for record in records
            )
            if not positives:
                continue
            aps, recalls = [], []
            for iou_threshold in IOU_THRESHOLDS:
                score_parts, tp_parts, ignore_parts = [], [], []
                for record in records:
                    scores, true_positive, detection_ignore = image_matches(
                        record, class_id, area_range, float(iou_threshold), max_det
                    )
                    score_parts.append(scores)
                    tp_parts.append(true_positive)
                    ignore_parts.append(detection_ignore)
                scores = np.concatenate(score_parts) if score_parts else np.empty(0)
                true_positive = np.concatenate(tp_parts) if tp_parts else np.empty(0, dtype=bool)
                detection_ignore = np.concatenate(ignore_parts) if ignore_parts else np.empty(0, dtype=bool)
                order = np.argsort(-scores, kind="stable")
                valid = ~detection_ignore[order]
                true_positive = true_positive[order][valid]
                false_positive = ~true_positive
                accumulated_tp = np.cumsum(true_positive)
                accumulated_fp = np.cumsum(false_positive)
                recall = accumulated_tp / positives
                precision = accumulated_tp / np.maximum(accumulated_tp + accumulated_fp, 1e-12)
                aps.append(interpolated_ap(recall, precision))
                recalls.append(float(recall[-1]) if len(recall) else 0.0)
            per_class[class_name] = {
                "ground_truth_instances": positives,
                "AP50_95": round(float(np.mean(aps)), 6),
                "AP50": round(aps[0], 6),
                "AR50_95": round(float(np.mean(recalls)), 6),
                "AR50": round(recalls[0], 6),
            }
        output[area_name] = {
            "ground_truth_instances": sum(item["ground_truth_instances"] for item in per_class.values()),
            "evaluated_classes": len(per_class),
            "AP50_95": round(float(np.mean([item["AP50_95"] for item in per_class.values()])), 6) if per_class else None,
            "AP50": round(float(np.mean([item["AP50"] for item in per_class.values()])), 6) if per_class else None,
            "AR50_95": round(float(np.mean([item["AR50_95"] for item in per_class.values()])), 6) if per_class else None,
            "AR50": round(float(np.mean([item["AR50"] for item in per_class.values()])), 6) if per_class else None,
            "per_class": per_class,
        }
    return output
