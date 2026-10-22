"""Render qualitative scene results (deployment-spec §16.2 item 8).

Draws GT (thin green) vs engine detections (colored, conf-labeled) on the 3
images per scene tag selected by prepare_frozen_subset.py, for both deployed
models (L0 and K1) side by side in the output summary.

  python3 scripts/render_scene_results.py \
      --engine-l0 engines/l0_fp16_b1.engine --engine-k1 engines/k1_fp16_b1.engine \
      --subset-dir data/frozen_subset --conf 0.25
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine-l0", type=Path, required=True)
    parser.add_argument("--engine-k1", type=Path, required=True)
    parser.add_argument("--subset-dir", type=Path, default=PACKAGE_ROOT / "data" / "frozen_subset")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--out", type=Path, default=PACKAGE_ROOT / "results" / "scene_renders")
    return parser.parse_args()


def draw(frame, boxes, scores, classes, names, gt=False):
    import cv2

    out = frame.copy()
    for box, score, cls in zip(boxes, scores, classes):
        x1, y1, x2, y2 = (int(v) for v in box)
        if gt:
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 0), 1)
            continue
        color = tuple(int(c) for c in ((37 * int(cls) + 60) % 255, (89 * int(cls) + 120) % 255, (167 * int(cls) + 40) % 255))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            out,
            f"{names.get(int(cls), int(cls))} {score:.2f}",
            (x1, max(12, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return out


def main() -> None:
    args = parse_args()
    import cv2

    from coco_eval_core import load_ground_truth_label
    from trt_runner import TrtEngine, decode_detections, load_input_image

    scene_path = args.subset_dir / "scene_manifest.json"
    if not scene_path.is_file():
        raise SystemExit(f"scene manifest missing: {scene_path} (run prepare_frozen_subset.py on the server)")
    scene_manifest = json.loads(scene_path.read_text(encoding="utf-8"))
    names = {int(k): v for k, v in json.loads((PACKAGE_ROOT / "data" / "bdd10k_classes.json").read_text(encoding="utf-8")).items()}

    engines = {"l0": TrtEngine(args.engine_l0, 1, args.imgsz), "k1": TrtEngine(args.engine_k1, 1, args.imgsz)}
    summary = {"conf": args.conf, "scenes": {}}
    for scene, entries in scene_manifest.items():
        scene_dir = args.out / scene
        scene_dir.mkdir(parents=True, exist_ok=True)
        summary["scenes"][scene] = []
        for entry in entries:
            name = entry["image"] if isinstance(entry, dict) else entry
            img_path = args.subset_dir / "images" / name
            img = cv2.imread(str(img_path))
            if img is None:
                raise SystemExit(f"cannot read {img_path}")
            h, w = img.shape[:2]
            gt = load_ground_truth_label(args.subset_dir / "labels" / (Path(name).stem + ".txt"), w, h)
            panels = {"gt": draw(img, gt[0], gt[1], gt[2], names, gt=True)}
            counts = {"gt": int(len(gt[0]))}
            for label, engine in engines.items():
                arr, params, orig_hw, _ = load_input_image(img_path)
                out = engine.infer(arr)["output0"]
                boxes, scores, classes = decode_detections(out, params, orig_hw, conf=args.conf)
                panels[label] = draw(img, boxes, scores, classes, names)
                counts[label] = int(len(boxes))
            combined = np.concatenate([panels["gt"], panels["l0"], panels["k1"]], axis=1)
            cv2.imwrite(str(scene_dir / name), combined)
            summary["scenes"][scene].append({"image": name, "detections": counts})
            print(f"[{scene}] {name}: gt={counts['gt']} l0={counts['l0']} k1={counts['k1']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary_path = args.out / "scene_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"[record] {summary_path}")
    print(f"[renders] {args.out}/<scene>/<image>.jpg  (GT | L0 | K1 three-panel strips)")


if __name__ == "__main__":
    main()
