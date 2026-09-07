"""Export Ultralytics detections in the official VisDrone DET text format."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset


def sha256(path: Path) -> str:
    """Return the SHA256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(paths: list[Path]) -> str:
    """Hash sorted filenames and file content into one deterministic tree digest."""
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.name):
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def format_detections(xyxy: np.ndarray, scores: np.ndarray, classes: np.ndarray, max_det: int = 500) -> str:
    """Convert zero-based XYXY detections to sorted VisDrone rows."""
    xyxy = np.asarray(xyxy, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    classes = np.asarray(classes, dtype=np.float64).reshape(-1)
    if not (len(xyxy) == len(scores) == len(classes)):
        raise ValueError("Detection boxes, scores, and classes must have equal lengths")
    if not (np.isfinite(xyxy).all() and np.isfinite(scores).all() and np.isfinite(classes).all()):
        raise ValueError("Detections contain NaN or Inf")
    rounded_classes = np.rint(classes).astype(np.int64)
    if not np.array_equal(classes, rounded_classes) or np.any((rounded_classes < 0) | (rounded_classes > 9)):
        raise ValueError("VisDrone export requires integer model classes in [0, 9]")

    order = np.argsort(-scores, kind="stable")[:max_det]
    rows = []
    for index in order:
        left, top, right, bottom = xyxy[index]
        width = max(0.0, right - left)
        height = max(0.0, bottom - top)
        rows.append(
            f"{left:.6f},{top:.6f},{width:.6f},{height:.6f},{scores[index]:.8f},"
            f"{rounded_classes[index] + 1},-1,-1"
        )
    return "\n".join(rows) + ("\n" if rows else "")


def resolve_source(data_path: Path, split: str) -> Path:
    """Resolve one image split through Ultralytics' dataset parser."""
    data = check_det_dataset(str(data_path.resolve()))
    source = data.get(split)
    if not isinstance(source, str):
        raise ValueError(f"VisDrone split must resolve to one directory, got {source!r}")
    path = Path(source)
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


def export(args: argparse.Namespace) -> dict:
    """Run prediction and write one official-format text file per image."""
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite export directory: {output}")
    output.mkdir(parents=True)

    model_path = args.model.resolve()
    model = YOLO(model_path)
    model_classes = sorted(int(index) for index in model.names)
    if model_classes != list(range(10)):
        raise ValueError(f"Model must have exactly VisDrone classes 0-9, got {model.names}")
    source = resolve_source(args.data, args.split)
    expected_images = {path.stem for path in source.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}}

    written = []
    detections = 0
    results = model.predict(
        source=str(source),
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        stream=True,
        verbose=False,
    )
    for result in results:
        boxes = result.boxes
        if boxes is None:
            text = ""
            count = 0
        else:
            text = format_detections(
                boxes.xyxy.detach().cpu().numpy(),
                boxes.conf.detach().cpu().numpy(),
                boxes.cls.detach().cpu().numpy(),
                args.max_det,
            )
            count = len(boxes)
        destination = output / f"{Path(result.path).stem}.txt"
        destination.write_text(text, encoding="ascii")
        written.append(destination)
        detections += min(count, args.max_det)

    actual_images = {path.stem for path in written}
    if actual_images != expected_images:
        missing = sorted(expected_images - actual_images)[:5]
        extra = sorted(actual_images - expected_images)[:5]
        raise RuntimeError(f"Prediction/image mismatch: missing={missing}, extra={extra}")

    evidence = {
        "format": "VisDrone DET: left,top,width,height,score,class(1-10),-1,-1",
        "model": str(model_path),
        "model_sha256": sha256(model_path),
        "data": str(args.data.resolve()),
        "split": args.split,
        "source": str(source),
        "images": len(written),
        "detections": detections,
        "settings": {
            "imgsz": args.imgsz,
            "batch": args.batch,
            "device": args.device,
            "conf": args.conf,
            "iou": args.iou,
            "max_det": args.max_det,
        },
        "output": str(output),
        "prediction_tree_sha256": tree_sha256(written),
    }
    manifest = output.parent / f"{output.name}.manifest.json"
    manifest.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    evidence["manifest"] = str(manifest)
    return evidence


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("configs/data/visdrone.yaml"))
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=500)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(export(parse_args()), indent=2))
