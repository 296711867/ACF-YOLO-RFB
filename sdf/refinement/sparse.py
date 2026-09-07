"""Geometry and fusion primitives for sparse high-resolution refinement."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def greedy_oracle_tiles(
    slice_boxes: Sequence[Sequence[int]],
    target_boxes: np.ndarray,
    top_k: int,
) -> tuple[list[int], float]:
    """Select tiles that greedily cover the most target-box centers.

    This helper deliberately consumes ground truth and is only an upper-bound
    diagnostic. It must never be used by the final inference method.
    """
    if top_k < 0:
        raise ValueError("top_k must be non-negative")
    targets = np.asarray(target_boxes, dtype=np.float32).reshape(-1, 4)
    if not len(slice_boxes) or top_k == 0:
        return [], 1.0 if not len(targets) else 0.0

    centers = (targets[:, :2] + targets[:, 2:]) / 2 if len(targets) else np.empty((0, 2), dtype=np.float32)
    covered = np.zeros(len(targets), dtype=bool)
    selected: list[int] = []
    remaining = set(range(len(slice_boxes)))
    for _ in range(min(top_k, len(slice_boxes))):
        best_index, best_gain = None, -1
        for index in sorted(remaining):
            x1, y1, x2, y2 = slice_boxes[index]
            inside = (
                (centers[:, 0] >= x1)
                & (centers[:, 0] < x2)
                & (centers[:, 1] >= y1)
                & (centers[:, 1] < y2)
            )
            gain = int(np.sum(inside & ~covered))
            if gain > best_gain:
                best_index, best_gain = index, gain
        if best_index is None:
            break
        selected.append(best_index)
        remaining.remove(best_index)
        x1, y1, x2, y2 = slice_boxes[best_index]
        covered |= (
            (centers[:, 0] >= x1)
            & (centers[:, 0] < x2)
            & (centers[:, 1] >= y1)
            & (centers[:, 1] < y2)
        )
    coverage = float(np.mean(covered)) if len(covered) else 1.0
    return selected, coverage


def merge_object_predictions(
    predictions: list,
    match_threshold: float = 0.5,
    match_metric: str = "IOS",
) -> list:
    """Apply the same class-aware GREEDY-NMM policy used by the SAHI baseline."""
    from sahi.postprocess.combine import GreedyNMMPostprocess

    postprocess = GreedyNMMPostprocess(
        match_threshold=match_threshold,
        match_metric=match_metric,
        class_agnostic=False,
    )
    return postprocess(predictions)


def shift_and_filter_local_predictions(predictions: Sequence, max_area: float) -> list:
    """Map crop predictions to the full image and retain small supplements only."""
    if max_area <= 0:
        raise ValueError("max_area must be positive")
    selected = []
    for prediction in predictions:
        shifted = prediction.get_shifted_object_prediction()
        x1, y1, x2, y2 = shifted.bbox.to_xyxy()
        if max(0.0, x2 - x1) * max(0.0, y2 - y1) < max_area:
            selected.append(shifted)
    return selected


def merge_scale_aware_predictions(
    full_predictions: Sequence,
    local_predictions: Sequence,
    max_local_area: float,
    match_threshold: float = 0.5,
) -> list:
    """Preserve full-frame medium/large boxes and fuse only small predictions."""
    if max_local_area <= 0:
        raise ValueError("max_local_area must be positive")
    full_large, full_small = [], []
    for prediction in full_predictions:
        x1, y1, x2, y2 = prediction.bbox.to_xyxy()
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        (full_small if area < max_local_area else full_large).append(prediction)
    fused_small = merge_object_predictions(list(full_small) + list(local_predictions), match_threshold)
    return full_large + fused_small


def object_predictions_to_arrays(predictions: Sequence) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert SAHI object predictions to evaluator-compatible NumPy arrays."""
    boxes, classes, scores = [], [], []
    for prediction in predictions:
        boxes.append(prediction.bbox.to_xyxy())
        classes.append(prediction.category.id)
        scores.append(prediction.score.value)
    return (
        np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
        np.asarray(classes, dtype=np.int64),
        np.asarray(scores, dtype=np.float32),
    )
