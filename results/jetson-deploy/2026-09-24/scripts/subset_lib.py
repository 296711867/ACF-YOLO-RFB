"""Shared helpers: frozen deterministic ordering and scene-tag selection.

The frozen subset rule (no RNG): order val filenames by md5(name) ascending
and take the first N. Scene tags use deterministic proxies computed from the
YOLO ground truth plus image brightness, because the converted BDD100K YOLO
labels do not carry the original BDD attributes:

  day    : mean grayscale brightness > 110
  night  : mean grayscale brightness < 60
  dense  : >= 30 GT boxes
  distant: >= 8 GT boxes and >= 60% of them with sqrt(area) < 24 px
  occlusion: >= 3 GT pairs with mutual IoU >= 0.30

Selection takes the 3 strongest images per scene tag (ties broken by name).
These thresholds are FROZEN with this package; do not retune them against
validation metrics.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

SCENES = ("day", "night", "dense", "distant", "occlusion")


def frozen_order(names: list[str]) -> list[str]:
    return sorted(names, key=lambda n: (hashlib.md5(n.encode("utf-8")).hexdigest(), n))


def gt_stats(gt_boxes: np.ndarray) -> dict:
    """Deterministic GT proxies for one image (xyxy pixel boxes)."""
    n = int(len(gt_boxes))
    if n == 0:
        return {"count": 0, "small_frac": 0.0, "overlap_pairs": 0}
    from coco_eval_core import box_iou

    sizes = np.sqrt(np.clip((gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1]), 0, None))
    ious = box_iou(gt_boxes, gt_boxes)
    np.fill_diagonal(ious, 0.0)
    overlap_pairs = int((np.triu(ious, 1) >= 0.30).sum())
    return {
        "count": n,
        "small_frac": float(np.mean(sizes < 24.0)),
        "overlap_pairs": overlap_pairs,
    }


def scene_tags(gt_boxes: np.ndarray, brightness: float) -> list[str]:
    tags = []
    if brightness > 110.0:
        tags.append("day")
    if brightness < 60.0:
        tags.append("night")
    stats = gt_stats(gt_boxes)
    if stats["count"] >= 30:
        tags.append("dense")
    if stats["count"] >= 8 and stats["small_frac"] >= 0.60:
        tags.append("distant")
    if stats["overlap_pairs"] >= 3:
        tags.append("occlusion")
    return tags


def image_brightness(image_bgr: np.ndarray) -> float:
    import cv2

    return float(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).mean())


def select_scenes(records: list[dict], per_scene: int = 3) -> dict[str, list[dict]]:
    """records: [{"image", "tags", "score_by_tag"}]; deterministic top-k per tag."""
    manifest: dict[str, list[dict]] = {}
    for scene in SCENES:
        candidates = [r for r in records if scene in r["tags"]]
        ranked = sorted(candidates, key=lambda r: (-r["score_by_tag"][scene], r["image"]))
        manifest[scene] = ranked[:per_scene]
    return manifest


def load_names(path: Path) -> dict[int, str]:
    return {int(k): v for k, v in json.loads(path.read_text(encoding="utf-8")).items()}
