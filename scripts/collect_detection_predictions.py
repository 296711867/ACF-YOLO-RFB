"""Stream frozen detector predictions to JSONL without metric aggregation."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--imgsz", type=int, required=True)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--resume", action="store_true", help="Verify and append to an interrupted JSONL prefix")
    args = parser.parse_args()
    metadata_path = args.output.with_suffix(".json")
    if metadata_path.exists() or (args.output.exists() and not args.resume):
        raise FileExistsError(f"refusing to overwrite prediction output: {args.output}")
    existing_images = []
    if args.output.exists():
        with args.output.open(encoding="utf-8") as existing_file:
            existing_images = [json.loads(line)["image"] for line in existing_file if line.strip()]
        if len(existing_images) != len(set(existing_images)):
            raise ValueError("resume output contains duplicate image paths")
    data = check_det_dataset(str(args.data.resolve()), autodownload=False, split=args.split)
    model = YOLO(args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    images = len(existing_images)
    results = model.predict(
        source=data[args.split],
        stream=True,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        verbose=False,
        save=False,
    )
    with args.output.open("a" if existing_images else "w", encoding="utf-8") as output_file:
        for result_index, result in enumerate(results):
            image_path = str(Path(result.path).resolve())
            if result_index < len(existing_images):
                if existing_images[result_index] != image_path:
                    raise ValueError(f"resume prefix differs at image {result_index + 1}")
                continue
            height, width = result.orig_shape
            boxes = result.boxes
            output_file.write(json.dumps({
                "image": image_path,
                "width": int(width),
                "height": int(height),
                "boxes_xyxy": boxes.xyxy.detach().cpu().tolist(),
                "classes": boxes.cls.detach().cpu().long().tolist(),
                "scores": boxes.conf.detach().cpu().tolist(),
            }) + "\n")
            images += 1
            if images % 500 == 0:
                print(f"predictions: {images}", flush=True)
    elapsed = time.perf_counter() - started
    metadata = {
        "status": "frozen predictions; no metric-based selection",
        "model": str(args.model.resolve()),
        "model_sha256": sha256(args.model),
        "data": str(args.data.resolve()),
        "data_sha256": sha256(args.data),
        "split": args.split,
        "images": images,
        "inference": {key: getattr(args, key) for key in ("imgsz", "batch", "device", "conf", "iou", "max_det")},
        "elapsed_seconds": elapsed,
        "resumed_from_images": len(existing_images),
        "predictions": str(args.output.resolve()),
        "predictions_sha256": sha256(args.output),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
