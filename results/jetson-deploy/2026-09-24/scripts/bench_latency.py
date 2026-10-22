"""Pure-model batch-1 latency benchmark (deployment-spec §16.2 item 2).

Protocol deliberately mirrors the frozen RTX-server §11.3 benchmark
(results/kd-arms/speed_trt_b1_l0_vs_k1.json): CUDA-event timing of pure
engine forwards with the H2D copy inside the timed region, warmup 48,
timed images 52 (positions [48:100] of the frozen 100-image list),
repeat 10 passes.

Two aggregations are reported:
  * pass-style (server-comparable): 10 passes x 52 images, ms per image;
  * per-forward distribution: every forward timed individually -> mean/std/
    min/P50/P95/P99/max, which is what §16.2 asks for (P50/P95 latency).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]

# Frozen RTX 3090 §11.3 reference (strongly-typed build on the all-float32
# ONNX, i.e. effectively an FP32 engine; see the deploy guide's precision
# note). Kept here only as context for the report - absolute latency is NOT
# comparable across RTX 3090 and Jetson.
RTX3090_REFERENCE = {
    "l0_ms_per_image_mean": 2.6527,
    "k1_ms_per_image_mean": 2.6449,
    "note": "RTX3090, TRT 11.3.0.99, strongly-typed engine on fp32 ONNX (fp32 execution); relative L0-vs-K1 parity only",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--label", default=None, help="model label, e.g. l0 or k1")
    parser.add_argument("--image-list", type=Path, default=PACKAGE_ROOT / "data" / "bench_images_100.txt")
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--warmup-images", type=int, default=48)
    parser.add_argument("--timed-images", type=int, default=52)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    idx = min(len(ordered) - 1, max(0, int(round(q / 100.0 * (len(ordered) - 1)))))
    return ordered[idx]


def main() -> None:
    args = parse_args()
    from trt_runner import TrtEngine, load_input_image as load_img

    names = [n for n in args.image_list.read_text().splitlines() if n.strip()]
    timed_names = names[args.warmup_images : args.warmup_images + args.timed_images]
    if len(timed_names) < args.timed_images:
        raise SystemExit(f"frozen list has only {len(timed_names)} timed images after warmup slice")
    engine = TrtEngine(args.engine, 1, args.imgsz)

    print(f"[load] decoding {len(timed_names)} timed images (+{args.warmup_images} warmup)...")
    warmup_arrs = [load_img(args.images_root / n)[0] for n in names[: args.warmup_images]]
    timed_arrs = [load_img(args.images_root / n)[0] for n in timed_names]

    # warmup
    for i in range(args.warmup_images):
        engine.forward(warmup_arrs[i % len(warmup_arrs)])
    engine.stream.synchronize()

    # pass-style timing (server-comparable)
    pass_ms = []
    for _ in range(args.repeat):
        engine.ev0.record(engine.stream)
        for arr in timed_arrs:
            engine.forward(arr)
        engine.ev1.record(engine.stream)
        engine.ev1.synchronize()
        pass_ms.append(engine.ev0.time_till(engine.ev1))

    # per-forward distribution
    per_forward_ms = []
    for _ in range(args.repeat):
        for arr in timed_arrs:
            per_forward_ms.append(engine.forward_timed_ms(arr))

    ms_per_image = [ms / len(timed_arrs) for ms in pass_ms]
    dist = {
        "n": len(per_forward_ms),
        "mean_ms": round(float(np.mean(per_forward_ms)), 4),
        "std_ms": round(float(np.std(per_forward_ms)), 4),
        "min_ms": round(min(per_forward_ms), 4),
        "p50_ms": round(percentile(per_forward_ms, 50), 4),
        "p95_ms": round(percentile(per_forward_ms, 95), 4),
        "p99_ms": round(percentile(per_forward_ms, 99), 4),
        "max_ms": round(max(per_forward_ms), 4),
    }
    label = args.label or args.engine.stem
    report = {
        "protocol": {
            "timing": "pure engine forward (H2D inside timed region), CUDA events",
            "batch": 1,
            "imgsz": args.imgsz,
            "warmup_forwards": args.warmup_images,
            "timed_images": len(timed_arrs),
            "repeat": args.repeat,
            "image_list": str(args.image_list),
        },
        "label": label,
        "engine": str(args.engine),
        "ms_per_run": [round(ms, 3) for ms in pass_ms],
        "ms_per_image": [round(ms, 4) for ms in ms_per_image],
        "ms_per_image_mean": round(float(np.mean(ms_per_image)), 4),
        "per_forward_distribution": dist,
        "rtx3090_reference": RTX3090_REFERENCE,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    out_path = args.output or (
        PACKAGE_ROOT / "results" / f"latency_b1_{label}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "label": label,
                "ms_per_image_mean": report["ms_per_image_mean"],
                "per_forward_distribution": dist,
            },
            indent=2,
        )
    )
    print(f"[record] {out_path}")


if __name__ == "__main__":
    main()
