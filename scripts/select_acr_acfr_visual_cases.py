"""Select ACR/ACFR visual-analysis cases from frozen per-image predictions (P20 §13).

Event rules (frozen at 2026-09-17, before rendering):
- Matching is class-aware greedy IoU >= 0.5 at prediction conf >= 0.25.
- acr_fixes_p2:   GT matched by ACR, missed by official-P2 (R0).
- acfr_fixes_acr: GT matched by ACFR, missed by ACR.
- acfr_new_fp:    ACFR prediction (conf >= 0.25) with no same-class GT at
                  IoU >= 0.5 and no ACR false positive within IoU >= 0.5.
- shared_miss:    GT missed by both ACR and ACFR.

Scenario proxies (BDD attribute metadata is not shipped in this repo, so
operational definitions are used and disclosed):
- distant_sign:  >= 1 traffic-sign GT with area <= 1024 px^2.
- night_light:   >= 1 traffic-light GT and mean gray luminance < 60/255.
- dense_vehicle: >= 15 car/truck/bus GT instances.
- occluded_ped:  >= 2 person GT with pairwise IoU >= 0.15 (overlap proxy).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["person", "rider", "car", "truck", "bus", "train", "motor", "bike", "traffic light", "traffic sign"]
VEHICLE_IDS = {2, 3, 4}
DISPLAY_CONF = 0.25
MATCH_IOU = 0.5

PREDS = {
    "p2": ROOT / "results/acr-stage20/r0-size-val-ep200.predictions.jsonl",
    "acr": ROOT / "results/acr-stage20/r1-size-val-ep200.predictions.jsonl",
    "acfr": ROOT / "results/acfr-stage20/a1-size-val-ep200.predictions.jsonl",
}
GT_DIR = ROOT / "data/BDD100K/labels/val"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-scenario", type=int, default=3)
    parser.add_argument("--luminance-cap", type=float, default=60.0)
    return parser.parse_args()


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    if not len(boxes_a) or not len(boxes_b):
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
    tl = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    br = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    inter = np.clip(br - tl, 0.0, None).prod(axis=2)
    area_a = np.clip(boxes_a[:, 2:] - boxes_a[:, :2], 0.0, None).prod(axis=1)
    area_b = np.clip(boxes_b[:, 2:] - boxes_b[:, :2], 0.0, None).prod(axis=1)
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-12)


def matched_gt_indices(pred_boxes, pred_classes, gt_boxes, gt_classes) -> set[int]:
    """Greedy class-aware IoU>=0.5 matching; returns matched GT indices."""
    matched: set[int] = set()
    if not len(pred_boxes) or not len(gt_boxes):
        return matched
    ious = iou_matrix(np.asarray(pred_boxes, dtype=np.float32), np.asarray(gt_boxes, dtype=np.float32))
    same_class = np.asarray(pred_classes)[:, None] == np.asarray(gt_classes)[None, :]
    ious = np.where(same_class, ious, 0.0)
    order = np.dstack(np.unravel_index(np.argsort(-ious, axis=None), ious.shape))[0]
    pred_used = set()
    for pi, gi in order:
        if ious[pi, gi] < MATCH_IOU:
            break
        if pi in pred_used or gi in matched:
            continue
        pred_used.add(pi)
        matched.add(int(gi))
    return matched


def load_jsonl(path: Path) -> dict[str, dict]:
    data = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        data[Path(record["image"]).name] = record
    return data


def load_gt(image_name: str, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    label_path = GT_DIR / (Path(image_name).stem + ".txt")
    boxes, classes = [], []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls, xc, yc, w, h = (float(x) for x in parts)
        boxes.append([(xc - w / 2) * width, (yc - h / 2) * height, (xc + w / 2) * width, (yc + h / 2) * height])
        classes.append(int(cls))
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4), np.asarray(classes, dtype=np.int64)


def filtered(record: dict) -> tuple[list, list]:
    keep = [i for i, score in enumerate(record["scores"]) if score >= DISPLAY_CONF]
    boxes = [record["boxes_xyxy"][i] for i in keep]
    classes = [record["classes"][i] for i in keep]
    return boxes, classes


def luminance(image_path: Path) -> float:
    image = cv2.imread(str(image_path))
    if image is None:
        return 255.0
    gray = cv2.cvtColor(cv2.resize(image, (160, 90)), cv2.COLOR_BGR2GRAY)
    return float(gray.mean())


def main() -> None:
    args = parse_args()
    preds = {name: load_jsonl(path) for name, path in PREDS.items()}
    image_names = sorted(set(preds["acfr"]) & set(preds["acr"]) & set(preds["p2"]))

    stats = {}
    for name in image_names:
        record = preds["acfr"][name]
        gt_boxes, gt_classes = load_gt(name, record["width"], record["height"])
        p2_boxes, p2_cls = filtered(preds["p2"][name])
        acr_boxes, acr_cls = filtered(preds["acr"][name])
        acfr_boxes, acfr_cls = filtered(preds["acfr"][name])

        matched = {
            "p2": matched_gt_indices(p2_boxes, p2_cls, gt_boxes, gt_classes),
            "acr": matched_gt_indices(acr_boxes, acr_cls, gt_boxes, gt_classes),
            "acfr": matched_gt_indices(acfr_boxes, acfr_cls, gt_boxes, gt_classes),
        }
        acr_fixes = matched["acr"] - matched["p2"]
        acfr_fixes = matched["acfr"] - matched["acr"]
        shared_miss = set(range(len(gt_classes))) - matched["acr"] - matched["acfr"]

        # ACFR false positives without a same-class GT match and without an ACR
        # false positive at the same location.
        acfr_fp_boxes = []
        for box, cls in zip(acfr_boxes, acfr_cls):
            ious = iou_matrix(np.asarray([box], dtype=np.float32), gt_boxes).ravel()
            same = [i for i, g in enumerate(gt_classes) if g == cls] if len(gt_classes) else []
            gt_match = any(ious[i] >= MATCH_IOU for i in same)
            if gt_match:
                continue
            if acr_boxes:
                acr_ious = iou_matrix(np.asarray([box], dtype=np.float32), np.asarray(acr_boxes, dtype=np.float32)).ravel()
                if any(score >= MATCH_IOU for score in acr_ious):
                    continue
            acfr_fp_boxes.append((box, cls))

        scenarios = {}
        areas = (gt_boxes[:, 2:] - gt_boxes[:, :2]).prod(axis=1) if len(gt_boxes) else np.asarray([])
        scenarios["distant_sign"] = bool(np.any([c == 9 for c in gt_classes]) and len(areas) and
                                         np.any([(gt_classes[i] == 9 and areas[i] <= 1024.0) for i in range(len(gt_classes))]))
        vehicle_count = int(np.sum(np.isin(gt_classes, list(VEHICLE_IDS))))
        scenarios["dense_vehicle"] = vehicle_count >= 15
        person_idx = [i for i, c in enumerate(gt_classes) if c == 0]
        occluded = False
        if len(person_idx) >= 2:
            person_ious = iou_matrix(gt_boxes[person_idx], gt_boxes[person_idx])
            np.fill_diagonal(person_ious, 0.0)
            occluded = bool((person_ious >= 0.15).any())
        scenarios["occluded_ped"] = occluded
        scenarios["night_light"] = False  # filled lazily below

        stats[name] = {
            "image": str(preds["acfr"][name]["image"]),
            "gt_count": int(len(gt_classes)),
            "acr_fixes_p2": len(acr_fixes),
            "acfr_fixes_acr": len(acfr_fixes),
            "acfr_new_fp": len(acfr_fp_boxes),
            "shared_miss": len(shared_miss),
            "has_tl_gt": any(c == 8 for c in gt_classes),
            "scenarios": scenarios,
            "_vehicle_count": vehicle_count,
            "_fix_total": len(acr_fixes) + len(acfr_fixes),
        }

    # Lazy luminance: only images with traffic-light GT need the night check.
    for name, entry in stats.items():
        entry["scenarios"]["night_light"] = bool(entry["has_tl_gt"] and luminance(Path(entry["image"])) < args.luminance_cap)
        entry.pop("has_tl_gt")

    def top_by(scenario: str, key: str, count: int) -> list[str]:
        pool = [n for n, e in stats.items() if e["scenarios"].get(scenario) and e[key] > 0]
        pool.sort(key=lambda n: (-stats[n][key], n))
        chosen = pool[:count]
        if len(chosen) < count:  # relax to any scenario, ranked the same way
            rest = [n for n, e in stats.items() if e[key] > 0 and n not in chosen]
            rest.sort(key=lambda n: (-stats[n][key], n))
            chosen += rest[: count - len(chosen)]
        return chosen

    def top_global(key: str, count: int) -> list[str]:
        pool = [n for n, e in stats.items() if e[key] > 0]
        pool.sort(key=lambda n: (-stats[n][key], n))
        return pool[:count]

    selection = {
        "five_column": {s: top_by(s, "_fix_total", args.per_scenario) for s in
                        ("distant_sign", "night_light", "dense_vehicle", "occluded_ped")},
        "acr_fixes_p2": top_global("acr_fixes_p2", 3),
        "acfr_fixes_acr": top_global("acfr_fixes_acr", 3),
        "acfr_new_fp": top_global("acfr_new_fp", 3),
        "shared_miss": top_global("shared_miss", 3),
    }

    totals = {
        "images_evaluated": len(image_names),
        "images_with_acr_fixes": sum(1 for e in stats.values() if e["acr_fixes_p2"] > 0),
        "images_with_acfr_fixes": sum(1 for e in stats.values() if e["acfr_fixes_acr"] > 0),
        "images_with_acfr_new_fp": sum(1 for e in stats.values() if e["acfr_new_fp"] > 0),
        "images_with_shared_miss": sum(1 for e in stats.values() if e["shared_miss"] > 0),
        "scenario_image_counts": {
            s: sum(1 for e in stats.values() if e["scenarios"][s])
            for s in ("distant_sign", "night_light", "dense_vehicle", "occluded_ped")
        },
    }

    output = {
        "work_package": "P20-ACR-ACFR-VISUALS-SELECTION",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rules": {
            "display_conf": DISPLAY_CONF,
            "match_iou": MATCH_IOU,
            "matching": "class-aware greedy IoU>=0.5 at conf>=0.25",
            "scenario_proxies": {
                "distant_sign": ">=1 traffic-sign GT with area<=1024 px^2",
                "night_light": ">=1 traffic-light GT and mean gray luminance<60/255",
                "dense_vehicle": ">=15 car/truck/bus GT",
                "occluded_ped": ">=2 person GT with pairwise IoU>=0.15",
            },
            "note": "BDD attribute metadata (time-of-day/scene) is not shipped in this repo; operational proxies above are frozen before rendering and disclosed in captions.",
        },
        "totals": totals,
        "selection": selection,
        "per_image": stats,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(totals, indent=2))
    print(json.dumps(selection, indent=2)[:1500])
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
