"""Numeric consistency check: FP16 engine vs FP32 engine on identical inputs.

Implements deployment-spec §16.1 item 4 (output shape, value range, NaN/Inf)
and bounds the FP16 quantization error before any accuracy evaluation.

CRITERION AMENDMENT (2026-09-24, pre-real-image rerun). The original raw
max|diff|<=5e-2 gate is unsuitable for an NMS-free top-k head, as diagnosed
on-device (NUMERIC_CONSISTENCY_DIAGNOSIS.md): below-threshold images have
unstable top-300 slot sets across precisions (whole-row misalignment), and
box columns are in 0-640 pixel units where FP16 quantization is legitimately
~0.5px. Gated criteria are now:

  G1  all outputs finite (no NaN/Inf);
  G2  score-column max|diff| <= 1e-2;
  G3  detection agreement >= 0.99 both directions at conf>=0.25
      (greedy IoU>=0.9 same-class match);
  G4  mean IoU of matched detections >= 0.95.

Raw whole-tensor max/mean abs diff is still REPORTED as a diagnostic but is
not gated. Default input is a deterministic synthetic batch (fixed seed, no
dataset dependency) - the formal run uses --images-root bench_100 street
frames. Synthetic input typically yields zero conf>=0.25 detections, making
G3/G4 trivially satisfied; it only serves as a NaN/Inf and score-column
smoke.

Suggested flow: synthetic first (sanity), then bench_100 images (formal).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine-fp16", type=Path, required=True)
    parser.add_argument("--engine-fp32", type=Path, required=True)
    parser.add_argument("--images-root", type=Path, default=None, help="optional: real images instead of synthetic")
    parser.add_argument("--image-list", type=Path, default=PACKAGE_ROOT / "data" / "bench_images_100.txt")
    parser.add_argument("--num", type=int, default=20, help="number of inputs to test")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=260923)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _match_agreement(a_boxes, a_cls, b_boxes, b_cls, iou_thr=0.9):
    """Greedy IoU matching of a-detections against b (same class).

    Returns (fraction matched, count, list of matched IoUs).
    """
    from coco_eval_core import box_iou

    if not len(a_boxes):
        return 1.0, 0, []
    matched = np.zeros(len(b_boxes), dtype=bool)
    hits, matched_ious = 0, []
    for i in range(len(a_boxes)):
        candidates = np.flatnonzero(b_cls == a_cls[i])
        candidates = candidates[~matched[candidates]]
        if not len(candidates):
            continue
        ious = box_iou(a_boxes[i][None], b_boxes[candidates])[0]
        j = candidates[int(np.argmax(ious))]
        if ious.max() >= iou_thr:
            matched[j] = True
            hits += 1
            matched_ious.append(float(ious.max()))
    return hits / len(a_boxes), len(a_boxes), matched_ious


def main() -> None:
    args = parse_args()
    from trt_runner import TrtEngine, decode_detections, load_input_image

    eng16 = TrtEngine(args.engine_fp16, 1, args.imgsz)
    eng32 = TrtEngine(args.engine_fp32, 1, args.imgsz)

    inputs = []
    if args.images_root is not None:
        names = [n for n in args.image_list.read_text().splitlines() if n.strip()][: args.num]
        for name in names:
            path = args.images_root / name
            arr, params, orig_hw, _ = load_input_image(path)
            inputs.append((path.name, arr, params, orig_hw))
        source = {"kind": "real_images", "list": str(args.image_list), "root": str(args.images_root)}
    else:
        rng = np.random.default_rng(args.seed)
        for i in range(args.num):
            arr = rng.random((1, 3, args.imgsz, args.imgsz), dtype=np.float32)
            inputs.append((f"synthetic_{i:03d}", arr, (1.0, 0, 0), (args.imgsz, args.imgsz)))
        source = {"kind": "synthetic_uniform", "seed": args.seed}

    per_image = []
    all_matched_ious: list[float] = []
    for name, arr, params, orig_hw in inputs:
        out16 = eng16.infer(arr)["output0"].reshape(-1, 6)
        out32 = eng32.infer(arr)["output0"].reshape(-1, 6)
        diff = np.abs(out16.astype(np.float64) - out32.astype(np.float64))
        score_max = float(diff[:, 4].max()) if diff.size else 0.0
        b16, s16, c16 = decode_detections(out16, params, orig_hw, conf=args.conf)
        b32, s32, c32 = decode_detections(out32, params, orig_hw, conf=args.conf)
        fwd_agr, n16, ious_fwd = _match_agreement(b16, c16, b32, c32)
        bwd_agr, n32, _ = _match_agreement(b32, c32, b16, c16)
        all_matched_ious.extend(ious_fwd)
        per_image.append(
            {
                "image": name,
                "shape": list(out16.shape),
                "nan16": int(np.isnan(out16).sum()),
                "inf16": int(np.isinf(out16).sum()),
                "score_max_abs_diff": score_max,
                "raw_max_abs_diff_diagnostic": float(diff.max()) if diff.size else 0.0,
                "raw_mean_abs_diff_diagnostic": float(diff.mean()) if diff.size else 0.0,
                "det16_count": n16,
                "det32_count": n32,
                "agreement_fwd": round(fwd_agr, 6),
                "agreement_bwd": round(bwd_agr, 6),
            }
        )
        print(
            f"{name}: score|d|max={score_max:.2e} raw|d|max={per_image[-1]['raw_max_abs_diff_diagnostic']:.3f} "
            f"dets {n16}/{n32} agr {fwd_agr:.4f}/{bwd_agr:.4f}"
        )

    nan_inf = sum(item["nan16"] + item["inf16"] for item in per_image)
    score_max = max(item["score_max_abs_diff"] for item in per_image)
    agreement = min(min(item["agreement_fwd"], item["agreement_bwd"]) for item in per_image)
    mean_matched_iou = float(np.mean(all_matched_ious)) if all_matched_ious else None
    verdict = {
        "G1_all_finite": nan_inf == 0,
        "G2_score_max_abs_diff": score_max,
        "G2_score_max_le_1e-2": score_max <= 1e-2,
        "G3_min_detection_agreement": agreement,
        "G3_agreement_ge_0p99": agreement >= 0.99,
        "G4_mean_matched_iou": mean_matched_iou,
        "G4_mean_matched_iou_ge_0p95": mean_matched_iou is None or mean_matched_iou >= 0.95,
        "raw_max_abs_diff_diagnostic": max(item["raw_max_abs_diff_diagnostic"] for item in per_image),
        "criterion_note": "raw tensor diff is diagnostic only (pixel-unit box columns, top-k slot instability); see header",
    }
    verdict["status"] = "PASS" if all(
        verdict[k] for k in ("G1_all_finite", "G2_score_max_le_1e-2", "G3_agreement_ge_0p99", "G4_mean_matched_iou_ge_0p95")
    ) else "FAIL"

    report = {
        "protocol": "FP16 vs FP32 engine on identical inputs; agreement = greedy IoU>=0.9 same-class match, conf>="
        + str(args.conf),
        "criterion_version": "2026-09-24 amendment (G1-G4); supersedes raw max|diff|<=5e-2",
        "engines": {
            "fp16": {"path": str(args.engine_fp16), "sha256": hashlib.sha256(args.engine_fp16.read_bytes()).hexdigest()},
            "fp32": {"path": str(args.engine_fp32), "sha256": hashlib.sha256(args.engine_fp32.read_bytes()).hexdigest()},
        },
        "input": source,
        "num_inputs": len(per_image),
        "verdict": verdict,
        "per_image": per_image,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    out_path = args.output or (PACKAGE_ROOT / "results" / "numeric_consistency.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(verdict, indent=2))
    print(f"[record] {out_path}")
    if verdict["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
