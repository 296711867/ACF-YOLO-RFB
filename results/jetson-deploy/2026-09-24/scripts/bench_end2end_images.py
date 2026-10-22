"""End-to-end image pipeline FPS benchmark (deployment-spec §16.2 item 3).

Measures the full single-frame path a deployment would run per image:
imdecode -> letterbox/normalize -> H2D -> TRT forward -> D2H -> conf filter
and box un-letterbox. Single-threaded by default (honest end-to-end number);
report mean / P50 / P95 per-frame latency and FPS after warmup.
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
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--label", default=None)
    parser.add_argument("--image-list", type=Path, default=PACKAGE_ROOT / "data" / "bench_images_100.txt")
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q / 100.0 * (len(ordered) - 1)))))
    return ordered[idx]


def main() -> None:
    args = parse_args()
    from trt_runner import TrtEngine, decode_detections, load_input_image

    names = [n for n in args.image_list.read_text().splitlines() if n.strip()]
    engine = TrtEngine(args.engine, 1, args.imgsz)

    # warmup on the first images (also caches file reads)
    for name in names[: args.warmup]:
        arr, _, _, _ = load_input_image(args.images_root / name)
        engine.infer(arr)

    total_ms, infer_ms, det_counts = [], [], []
    for name in names:
        t0 = time.perf_counter()
        arr, params, orig_hw, _ = load_input_image(args.images_root / name)
        t1 = time.perf_counter()
        out = engine.infer(arr)["output0"]
        t2 = time.perf_counter()
        boxes, scores, classes = decode_detections(out, params, orig_hw, conf=args.conf)
        t3 = time.perf_counter()
        total_ms.append((t3 - t0) * 1000.0)
        infer_ms.append((t2 - t1) * 1000.0)
        det_counts.append(int(len(boxes)))

    mean_ms = float(np.mean(total_ms))
    label = args.label or args.engine.stem
    report = {
        "protocol": {
            "pipeline": "imread + letterbox640 + H2D + TRT forward + D2H + conf-filter postprocess",
            "batch": 1,
            "imgsz": args.imgsz,
            "conf": args.conf,
            "warmup_images": args.warmup,
            "timed_images": len(names),
            "image_list": str(args.image_list),
            "threads": 1,
        },
        "label": label,
        "engine": str(args.engine),
        "total_ms": {
            "mean": round(mean_ms, 3),
            "p50": round(percentile(total_ms, 50), 3),
            "p95": round(percentile(total_ms, 95), 3),
        },
        "infer_only_ms_mean": round(float(np.mean(infer_ms)), 3),
        "fps_mean": round(1000.0 / mean_ms, 3),
        "mean_detections_per_image": round(float(np.mean(det_counts)), 2),
        "per_image_ms": [round(ms, 3) for ms in total_ms],
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    out_path = args.output or (
        PACKAGE_ROOT / "results" / f"end2end_images_{label}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("label", "total_ms", "infer_only_ms_mean", "fps_mean")}, indent=2))
    print(f"[record] {out_path}")


if __name__ == "__main__":
    main()
