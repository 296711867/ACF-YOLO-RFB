"""Engineering inference-speed benchmark: candidate vs baseline on one clean GPU.

Measures FP32 batch-N end-to-end predict latency over a fixed slice of BDD val
images with warmup, following the project's honest-cost reporting discipline.
Historical invocations used --acr/--official (labels acr/official_p2); the
--candidate/--baseline aliases plus label args generalize to any pair.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--acr", "--candidate", dest="candidate", type=Path, required=True)
    parser.add_argument("--official", "--baseline", dest="baseline", type=Path, required=True)
    parser.add_argument("--candidate-label", default="acr")
    parser.add_argument("--baseline-label", default="official_p2")
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
                            "repeat": args.repeat,
                            "image_list": str(args.image_list),
                            "image_list_rule": "sorted(data/BDD100K/images/val) first 100"}}
    delta_key = f"{args.candidate_label}_vs_{args.baseline_label}_ms_delta_percent"
    for label, weight in ((args.baseline_label, args.baseline), (args.candidate_label, args.candidate)):
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
    delta = results[args.candidate_label]["ms_per_image_mean"] / results[args.baseline_label]["ms_per_image_mean"] - 1
    results[delta_key] = round(delta * 100, 2)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: results[k] for k in (args.baseline_label, args.candidate_label, delta_key)}, indent=2))


if __name__ == "__main__":
    main()
