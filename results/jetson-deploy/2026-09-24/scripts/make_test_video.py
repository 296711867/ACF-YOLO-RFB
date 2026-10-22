"""Build a deterministic test video from the frozen subset images.

Writes an image-sequence mp4 (BDD stills in frozen-subset order) used by
bench_end2end_video.py. The video is for throughput/latency measurement only
- BDD frames are independent stills, so no temporal accuracy claims.

Frames are letterboxed to the target size (BDD is natively 1280x720, so the
default is a straight copy).
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
    parser.add_argument("--images-root", type=Path, required=True, help="directory containing the subset jpgs")
    parser.add_argument("--manifest", type=Path, default=None, help="subset_manifest.json for frozen ordering")
    parser.add_argument("--frames", type=int, default=450, help="15 s at 30 fps")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import cv2

    if args.manifest is not None:
        names = [entry["image"] for entry in json.loads(args.manifest.read_text(encoding="utf-8"))["files"]]
    else:
        names = sorted(p.name for p in args.images_root.glob("*.jpg"))
    names = [n for n in names if n.lower().endswith(".jpg")][: args.frames]
    if len(names) < args.frames:
        print(f"[warn] only {len(names)} images available, video will be shorter than requested")

    out_path = args.output or (PACKAGE_ROOT / "data" / f"test_video_{args.width}x{args.height}.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (args.width, args.height))

    canvas = np.full((args.height, args.width, 3), 114, dtype=np.uint8)
    for name in names:
        img = cv2.imread(str(args.images_root / name))
        if img is None:
            raise SystemExit(f"cannot read {name}")
        h, w = img.shape[:2]
        r = min(args.width / w, args.height / h)
        nh, nw = int(round(h * r)), int(round(w * r))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        frame = canvas.copy()
        top, left = (args.height - nh) // 2, (args.width - nw) // 2
        frame[top : top + nh, left : left + nw] = resized
        writer.write(frame)
    writer.release()

    meta = {
        "video": str(out_path),
        "frames": len(names),
        "fps": args.fps,
        "size": [args.width, args.height],
        "duration_s": round(len(names) / args.fps, 2),
        "source": {"images_root": str(args.images_root), "manifest": str(args.manifest) if args.manifest else None},
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    meta_path = out_path.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
