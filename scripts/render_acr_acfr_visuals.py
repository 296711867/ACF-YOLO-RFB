"""Render ACR/ACFR paper visuals from frozen weights (P20 §13).

Reads the selection manifest written by select_acr_acfr_visual_cases.py and
produces: five-column GT|YOLO26n|P2|ACR|ACFR panels per scenario, event
highlight panels (ACR fixes P2, ACFR fixes ACR, ACFR new false positives,
shared misses), and ACFR G=4 gating heatmaps. CPU inference only.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from select_acr_acfr_visual_cases import (  # noqa: E402
    CLASSES,
    DISPLAY_CONF,
    MATCH_IOU,
    filtered,
    iou_matrix,
    load_gt,
    load_jsonl,
    matched_gt_indices,
)

MODELS = {
    "YOLO26n": ROOT / "runs/detect/phase1/bdd100k-yolo26n-640-e200-seed0/weights/best.pt",
    "P2": ROOT / "runs/detect/phase1/bdd100k-yolo26n-p2-640-e200-seed0/weights/best.pt",
    "ACR": ROOT / "runs/detect/phase18b-acr-stage20/bdd100k-yolo26n-p2-acr-r1-candidate-640-e200-stop20-b32-seed0/weights/best.pt",
    "ACR+ACFR": ROOT / "runs/detect/phase19a-acfr-stage20/bdd100k-yolo26n-p2-acr-acfr-a1-640-e200-stop20-b32-seed0/weights/best.pt",
}
PREDS = {
    "P2": ROOT / "results/acr-stage20/r0-size-val-ep200.predictions.jsonl",
    "ACR": ROOT / "results/acr-stage20/r1-size-val-ep200.predictions.jsonl",
    "ACFR": ROOT / "results/acfr-stage20/a1-size-val-ep200.predictions.jsonl",
}
SCENARIO_TITLES = {
    "distant_sign": "distant traffic signs",
    "night_light": "night traffic lights",
    "dense_vehicle": "dense vehicles",
    "occluded_ped": "occluded pedestrians",
}
COLORS = [
    (230, 57, 70), (29, 53, 87), (42, 157, 143), (231, 111, 81),
    (94, 96, 206), (188, 71, 73), (48, 105, 136), (155, 93, 229),
    (255, 191, 0), (0, 176, 240),
]
HIGHLIGHT = (0, 255, 255)  # yellow for event boxes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def draw_boxes(image: np.ndarray, boxes, classes, scores=None, highlight=(), thickness=2) -> np.ndarray:
    canvas = image.copy()
    for index, (box, cls) in enumerate(zip(boxes, classes)):
        color = HIGHLIGHT if index in highlight else COLORS[int(cls) % len(COLORS)]
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness if index not in highlight else thickness + 2)
        label = CLASSES[int(cls)]
        if scores is not None:
            label = f"{label} {float(scores[index]):.2f}"
        if index in highlight:
            scale, t = 0.45, 1
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, t)
            cv2.rectangle(canvas, (x1, max(0, y1 - th - 4)), (x1 + tw + 2, y1), color, -1)
            cv2.putText(canvas, label, (x1, max(10, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), t, cv2.LINE_AA)
    return canvas


def letterbox640(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(640 / width, 640 / height)
    new_w, new_h = round(width * scale), round(height * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
    top, left = (640 - new_h) // 2, (640 - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas, scale, top, left


def predict(yolo: YOLO, image_path: Path):
    result = yolo(str(image_path), imgsz=640, conf=DISPLAY_CONF, device="cpu", verbose=False)[0]
    boxes = result.boxes.xyxy.cpu().numpy().tolist() if len(result.boxes) else []
    classes = result.boxes.cls.cpu().numpy().astype(int).tolist() if len(result.boxes) else []
    scores = result.boxes.conf.cpu().numpy().tolist() if len(result.boxes) else []
    return boxes, classes, scores


def compose_row(panels: list[tuple[str, np.ndarray]], out_path: Path, dpi: int = 300) -> None:
    fig, axes = plt.subplots(1, len(panels), figsize=(3.1 * len(panels), 3.4))
    for ax, (title, image) in zip(np.atleast_1d(axes), panels):
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.tight_layout(pad=0.4)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "five_col").mkdir(exist_ok=True)
    (args.out_dir / "events").mkdir(exist_ok=True)
    (args.out_dir / "gate_heatmaps").mkdir(exist_ok=True)

    frozen_preds = {name: load_jsonl(path) for name, path in PREDS.items()}
    models = {name: YOLO(str(path)) for name, path in MODELS.items()}

    image_index = {}
    for record in frozen_preds["ACFR"].values():
        image_index[Path(record["image"]).name] = record

    def image_path_of(name: str) -> Path:
        return Path(image_index[name]["image"])

    def gt_of(name: str):
        record = image_index[name]
        return load_gt(name, record["width"], record["height"])

    def frozen(name: str, model: str):
        return filtered(frozen_preds[model][name])

    cache: dict[str, dict] = {}

    def render_predictions(name: str) -> dict:
        if name in cache:
            return cache[name]
        path = image_path_of(name)
        image = cv2.imread(str(path))
        gt_boxes, gt_classes = gt_of(name)
        entry = {"image": image, "gt_boxes": gt_boxes, "gt_classes": gt_classes}
        for label in MODELS:
            if label == "YOLO26n":
                entry[label] = predict(models[label], path)
            else:
                key = "ACFR" if label == "ACR+ACFR" else label
                boxes, classes = frozen(name, key)
                scores = [s for s, keep in zip(frozen_preds[key][name]["scores"], range(len(boxes)))]
                entry[label] = (boxes, classes, scores)
        cache[name] = entry
        return entry

    # ---- five-column panels per scenario ----
    rendered_log = []
    for scenario, names in manifest["selection"]["five_column"].items():
        for index, name in enumerate(names, 1):
            entry = render_predictions(name)
            panels = [("GT", draw_boxes(entry["image"], entry["gt_boxes"], entry["gt_classes"]))]
            for label in MODELS:
                boxes, classes, scores = entry[label]
                panels.append((label, draw_boxes(entry["image"], boxes, classes, scores)))
            out = args.out_dir / "five_col" / f"{scenario}_{index}_{Path(name).stem}.png"
            compose_row(panels, out)
            rendered_log.append(str(out))

    # ---- event highlight panels ----
    def event_panel(names: list[str], column_a: str, column_b: str, out_prefix: str, mode: str) -> None:
        for index, name in enumerate(names, 1):
            entry = render_predictions(name)
            gt_boxes, gt_classes = entry["gt_boxes"], entry["gt_classes"]
            boxes = {label: entry[label] for label in MODELS}
            key_a = "ACFR" if column_a == "ACR+ACFR" else column_a
            key_b = "ACFR" if column_b == "ACR+ACFR" else column_b
            matched_a = matched_gt_indices(*frozen(name, key_a), gt_boxes, gt_classes)
            matched_b = matched_gt_indices(*frozen(name, key_b), gt_boxes, gt_classes)

            if mode == "fix":  # column_b fixes what column_a missed
                events = matched_b - matched_a
                gt_highlight = list(events)
                panel_a = draw_boxes(entry["image"], gt_boxes, gt_classes, highlight=gt_highlight)
                box_b, cls_b, score_b = entry[column_b]
                pred_highlight = []
                for pi, (box, cls) in enumerate(zip(box_b, cls_b)):
                    ious = iou_matrix(np.asarray([box], dtype=np.float32), gt_boxes).ravel() if len(gt_boxes) else []
                    for gi in events:
                        if gi < len(ious) and ious[gi] >= MATCH_IOU and gt_classes[gi] == cls:
                            pred_highlight.append(pi)
                            break
                panel_b = draw_boxes(entry["image"], box_b, cls_b, score_b, highlight=pred_highlight)
                panels = [(f"GT ({len(events)} event boxes)", panel_a), (column_a, panel_a), (column_b, panel_b)]
            else:  # shared miss: highlight GT missed by both
                events = set(range(len(gt_classes))) - matched_a - matched_b
                panel = draw_boxes(entry["image"], gt_boxes, gt_classes, highlight=list(events))
                panels = [(f"GT ({len(events)} missed)", panel), (column_a, panel), (column_b, panel)]

            out = args.out_dir / "events" / f"{out_prefix}_{index}_{Path(name).stem}.png"
            compose_row(panels, out)
            rendered_log.append(str(out))

    event_panel(manifest["selection"]["acr_fixes_p2"], "P2", "ACR", "acr_fixes_p2", "fix")
    event_panel(manifest["selection"]["acfr_fixes_acr"], "ACR", "ACR+ACFR", "acfr_fixes_acr", "fix")

    # ACFR new false positives: highlight ACFR predictions without GT match.
    for index, name in enumerate(manifest["selection"]["acfr_new_fp"], 1):
        entry = render_predictions(name)
        gt_boxes, gt_classes = entry["gt_boxes"], entry["gt_classes"]
        box_b, cls_b, score_b = entry["ACR+ACFR"]
        highlight = []
        for pi, (box, cls) in enumerate(zip(box_b, cls_b)):
            ious = iou_matrix(np.asarray([box], dtype=np.float32), gt_boxes).ravel() if len(gt_boxes) else []
            same = [i for i, g in enumerate(gt_classes) if g == cls]
            if not any(i < len(ious) and ious[i] >= MATCH_IOU for i in same):
                highlight.append(pi)
        panel_gt = draw_boxes(entry["image"], gt_boxes, gt_classes)
        panel_fp = draw_boxes(entry["image"], box_b, cls_b, score_b, highlight=highlight)
        out = args.out_dir / "events" / f"acfr_new_fp_{index}_{Path(name).stem}.png"
        compose_row([("GT", panel_gt), ("ACR", draw_boxes(entry["image"], *entry["ACR"][:2], entry["ACR"][2])), ("ACR+ACFR (FP highlighted)", panel_fp)], out)
        rendered_log.append(str(out))

    event_panel(manifest["selection"]["shared_miss"], "ACR", "ACR+ACFR", "shared_miss", "miss")

    # ---- ACFR G=4 gating heatmaps ----
    acfr_yolo = models["ACR+ACFR"]
    gate_module = acfr_yolo.model.model[18].gate
    captured: dict[str, np.ndarray] = {}

    def gate_hook(_module, _inputs, output):
        captured["logits"] = output.detach().float().cpu().numpy()

    handle = gate_module.register_forward_hook(gate_hook)
    gate_images = [
        manifest["selection"]["five_column"]["dense_vehicle"][0],
        manifest["selection"]["five_column"]["distant_sign"][1],
    ]
    for name in gate_images:
        path = image_path_of(name)
        acfr_yolo(str(path), imgsz=640, conf=DISPLAY_CONF, device="cpu", verbose=False)
        logits = captured["logits"][0]  # [4, H, W]
        weights = 1.0 / (1.0 + np.exp(-logits))
        image = cv2.imread(str(path))
        canvas, scale, top, left = letterbox640(image)
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 7.6))
        for group, ax in enumerate(axes.ravel()):
            heat = cv2.resize(weights[group], (640, 640), interpolation=cv2.INTER_LINEAR)
            overlay = canvas.copy()
            heat_color = cv2.applyColorMap(np.clip(heat * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
            mask = cv2.resize((cv2.resize(weights[group], (640, 640)) > 0).astype(np.uint8), (640, 640))
            blended = cv2.addWeighted(overlay, 0.55, heat_color, 0.45, 0)
            blended[mask == 0] = overlay[mask == 0]
            ax.imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
            ax.set_title(f"gate group G{group + 1} (gain for P3 path)", fontsize=9)
            ax.axis("off")
        fig.suptitle(f"ACFR complementary gate weights — {name}", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        out = args.out_dir / "gate_heatmaps" / f"gate_g4_{Path(name).stem}.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
        rendered_log.append(str(out))
    handle.remove()

    log = {
        "work_package": "P20-ACR-ACFR-VISUALS-RENDER",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "figures": rendered_log,
        "models": {name: str(path.relative_to(ROOT)) for name, path in MODELS.items()},
        "inference": "YOLO26n column: fresh CPU predict conf>=0.25; P2/ACR/ACFR columns: frozen eval predictions filtered to conf>=0.25",
    }
    (args.out_dir / "render_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"rendered {len(rendered_log)} figures -> {args.out_dir}")


if __name__ == "__main__":
    main()
