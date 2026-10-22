"""End-to-end video FPS benchmark (deployment-spec §16.2 item 4).

Streams a video file frame by frame through the same single-frame pipeline
as bench_end2end_images.py (read -> letterbox -> TRT forward -> postprocess),
reports steady-state FPS and per-frame end-to-end P50/P95 after a fixed
warmup window. Optionally writes an annotated video for qualitative checks.

The BDD100K frames are independent stills, so the synthetic test video built
by make_test_video.py is an *image-sequence* video: valid for throughput and
latency measurement, not for tracking-style accuracy claims. State that in
the thesis when reporting this number.
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
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--warmup-frames", type=int, default=50)
    parser.add_argument("--max-frames", type=int, default=0, help="0 = all frames")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--draw-out", type=Path, default=None, help="optionally write an annotated mp4")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q / 100.0 * (len(ordered) - 1)))))
    return ordered[idx]


def draw_boxes(frame, boxes, scores, classes, names):
    import cv2

    out = frame.copy()
    for (x1, y1, x2, y2), score, cls in zip(boxes, scores, classes):
        color = tuple(int(c) for c in ((37 * int(cls) + 60) % 255, (89 * int(cls) + 120) % 255, (167 * int(cls) + 40) % 255))
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(
            out,
            f"{names.get(int(cls), int(cls))} {score:.2f}",
            (int(x1), max(12, int(y1) - 4)),
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

    from trt_runner import TrtEngine, decode_detections, preprocess

    engine = TrtEngine(args.engine, 1, args.imgsz)
    names = json.loads((PACKAGE_ROOT / "data" / "bdd10k_classes.json").read_text(encoding="utf-8"))

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {args.video}")
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.draw_out is not None:
        args.draw_out.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(args.draw_out), cv2.VideoWriter_fourcc(*"mp4v"), fps_src, (width, height)
        )

    per_frame_ms, frame_index, det_counts = [], 0, []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_index += 1
        if args.max_frames and frame_index > args.max_frames:
            break
        t0 = time.perf_counter()
        arr, params = preprocess(frame, args.imgsz)
        out = engine.infer(arr)["output0"]
        boxes, scores, classes = decode_detections(out, params, frame.shape[:2], conf=args.conf)
        if writer is not None:
            writer.write(draw_boxes(frame, boxes, scores, classes, names))
        t1 = time.perf_counter()
        det_counts.append(int(len(boxes)))
        if frame_index > args.warmup_frames:
            per_frame_ms.append((t1 - t0) * 1000.0)

    cap.release()
    if writer is not None:
        writer.release()

    if not per_frame_ms:
        raise SystemExit("no steady-state frames measured; video too short for the warmup window")

    mean_ms = float(np.mean(per_frame_ms))
    label = args.label or args.engine.stem
    report = {
        "protocol": {
            "pipeline": "video read + letterbox640 + TRT forward + postprocess (single thread)",
            "warmup_frames_excluded": args.warmup_frames,
            "steady_frames_measured": len(per_frame_ms),
            "conf": args.conf,
            "source_fps": round(fps_src, 3),
            "resolution": [width, height],
            "video": str(args.video),
            "note": "image-sequence video (BDD stills) - throughput/latency metric only, not tracking accuracy",
        },
        "label": label,
        "engine": str(args.engine),
        "steady_total_ms": {
            "mean": round(mean_ms, 3),
            "p50": round(percentile(per_frame_ms, 50), 3),
            "p95": round(percentile(per_frame_ms, 95), 3),
        },
        "steady_fps_mean": round(1000.0 / mean_ms, 3),
        "realtime_factor": round((1000.0 / mean_ms) / fps_src, 3),
        "mean_detections_per_frame": round(float(np.mean(det_counts)), 2),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    out_path = args.output or (
        PACKAGE_ROOT / "results" / f"end2end_video_{label}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {k: report[k] for k in ("label", "steady_total_ms", "steady_fps_mean", "realtime_factor")}, indent=2
        )
    )
    print(f"[record] {out_path}")


if __name__ == "__main__":
    main()
