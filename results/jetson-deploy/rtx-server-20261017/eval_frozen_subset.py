"""Frozen-subset AP consistency evaluation (deployment-spec §16.1 item 5).

Runs a TensorRT engine over the frozen BDD100K subset (prepared by
prepare_frozen_subset.py) with the frozen evaluation protocol
(conf 0.001, maxDet 300, letterbox 640) and computes COCO-style metrics with
the IDENTICAL core as the thesis 10k numbers (see coco_eval_core.py header).

Run it on BOTH sides and compare:
  RTX server reference : python3 scripts/eval_frozen_subset.py --engine engines/l0_fp32_b1.engine --subset-dir data/frozen_subset --label l0_rtx_fp32
  Jetson deployment    : python3 scripts/eval_frozen_subset.py --engine engines/l0_fp16_b1.engine --subset-dir data/frozen_subset --label l0_jetson_fp16

Consistency is judged on the DELTA between the two runs on the same subset;
absolute subset AP is a subset-level statistic, not the thesis 10k number.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--subset-dir", type=Path, default=PACKAGE_ROOT / "data" / "frozen_subset")
    parser.add_argument("--label", required=True, help="e.g. l0_jetson_fp16")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from coco_eval_core import AREA_RANGES, evaluate, load_ground_truth_label, make_record
    from trt_runner import TrtEngine, decode_detections, load_input_image

    images_dir = args.subset_dir / "images"
    labels_dir = args.subset_dir / "labels"
    names = {int(k): v for k, v in json.loads((PACKAGE_ROOT / "data" / "bdd10k_classes.json").read_text(encoding="utf-8")).items()}
    manifest_path = args.subset_dir / "subset_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        names_files = [e["image"] for e in manifest["files"] if e["role"] == "subset"]
    else:
        names_files = sorted(p.name for p in images_dir.glob("*.jpg"))
    if not names_files:
        raise SystemExit(f"no subset images under {images_dir}; run prepare_frozen_subset.py on the server first")

    engine = TrtEngine(args.engine, args.batch, args.imgsz)
    records, predictions = [], []
    t0 = time.perf_counter()
    for index, name in enumerate(names_files):
        arr, params, orig_hw, _ = load_input_image(images_dir / name)
        out = engine.infer(arr)["output0"]
        boxes, scores, classes = decode_detections(out, params, orig_hw, conf=args.conf)
        gt = load_ground_truth_label(labels_dir / (Path(name).stem + ".txt"), orig_hw[1], orig_hw[0])
        records.append(make_record(gt, boxes, classes, scores))
        predictions.append(
            {
                "image": name,
                "width": orig_hw[1],
                "height": orig_hw[0],
                "boxes_xyxy": boxes.tolist(),
                "classes": classes.tolist(),
                "scores": scores.tolist(),
            }
        )
        if (index + 1) % 100 == 0:
            print(f"  {index + 1}/{len(names_files)} images, {(time.perf_counter() - t0):.0f}s elapsed", flush=True)

    metrics = evaluate(records, names, args.max_det, progress=True)
    report = {
        "protocol": "COCO-style 101-point AP, IoU 0.50:0.95, maxDet 300 (identical core as scripts/evaluate_object_sizes.py)",
        "area_ranges_pixels": {
            key: [value[0], value[1] if math.isfinite(value[1]) else None] for key, value in AREA_RANGES.items()
        },
        "label": args.label,
        "engine": str(args.engine),
        "engine_sha256": hashlib.sha256(args.engine.read_bytes()).hexdigest(),
        "subset_dir": str(args.subset_dir),
        "images": len(records),
        "inference": {"imgsz": args.imgsz, "batch": args.batch, "conf": args.conf, "max_det": args.max_det},
        "wall_seconds": round(time.perf_counter() - t0, 1),
        "metrics": metrics,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    out_path = args.output or (PACKAGE_ROOT / "results" / f"subset_size_val_{args.label}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    pred_path = out_path.with_suffix(".predictions.jsonl")
    with pred_path.open("w", encoding="utf-8") as f:
        for item in predictions:
            f.write(json.dumps(item) + "\n")
    print(
        json.dumps(
            {
                "label": args.label,
                "images": len(records),
                "all_AP50_95": metrics["all"]["AP50_95"],
                "small_AP50_95": metrics["small"]["AP50_95"],
                "small_AR50_95": metrics["small"]["AR50_95"],
            },
            indent=2,
        )
    )
    print(f"[record] {out_path}\n[predictions] {pred_path}")


if __name__ == "__main__":
    main()
