"""Engineering inference-speed benchmark: ACR vs official P2 on one clean GPU.

Measures FP32 batch-16 end-to-end predict latency over a fixed slice of BDD val
images with warmup, following the project's honest-cost reporting discipline.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--acr", type=Path, required=True)
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--image-list", type=Path, required=True)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--warmup-images", type=int, default=48)
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()

    names = [line.strip() for line in args.image_list.read_text().splitlines() if line.strip()]
    paths = [str((args.images_root / name).resolve()) for name in names]
    warmup, timed = paths[: args.warmup_images], paths[args.warmup_images:]

    results = {"protocol": {"device": args.device, "batch": args.batch, "fp32": True,
                            "warmup_images": args.warmup_images, "timed_images": len(timed),
                            "repeat": args.repeat}}
    for label, weight in (("official_p2", args.official), ("acr", args.acr)):
        model = YOLO(str(weight))
        model.predict(warmup, imgsz=640, device=args.device, batch=args.batch, verbose=False, half=False)
        runs = []
        for _ in range(args.repeat):
            start = time.perf_counter()
            model.predict(timed, imgsz=640, device=args.device, batch=args.batch, verbose=False, half=False)
            runs.append(time.perf_counter() - start)
        per_image_ms = [run * 1000 / len(timed) for run in runs]
        params = sum(p.numel() for p in model.model.parameters())
        results[label] = {"weight": str(weight), "params": params,
                          "total_seconds_per_run": [round(r, 4) for r in runs],
                          "ms_per_image": [round(v, 3) for v in per_image_ms],
                          "ms_per_image_mean": round(sum(per_image_ms) / len(per_image_ms), 3)}
    delta = results["acr"]["ms_per_image_mean"] / results["official_p2"]["ms_per_image_mean"] - 1
    results["acr_vs_official_ms_delta_percent"] = round(delta * 100, 2)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: results[k] for k in ("official_p2", "acr", "acr_vs_official_ms_delta_percent")}, indent=2))


if __name__ == "__main__":
    main()
