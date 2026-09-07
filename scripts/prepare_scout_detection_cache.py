"""Convert frozen scout predictions and utility labels into set-valued caches."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


TOKEN_FEATURES = ("cx", "cy", "width", "height", "confidence", "sqrt_area", "log_aspect_ratio")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def encode_detection_tokens(
    prediction: dict,
    max_tokens: int = 192,
    num_classes: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode one prediction record using stable score order and normalized geometry."""
    if max_tokens <= 0 or num_classes <= 0:
        raise ValueError("max_tokens and num_classes must be positive")
    width, height = float(prediction["width"]), float(prediction["height"])
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    boxes = np.asarray(prediction["boxes_xyxy"], dtype=np.float32).reshape(-1, 4)
    classes = np.asarray(prediction["classes"], dtype=np.int64)
    scores = np.asarray(prediction["scores"], dtype=np.float32)
    if len(boxes) != len(classes) or len(boxes) != len(scores):
        raise ValueError("prediction boxes, classes, and scores differ in length")
    if len(classes) and (classes.min() < 0 or classes.max() >= num_classes):
        raise ValueError("prediction class is outside the configured range")
    order = np.argsort(-scores, kind="stable")[:max_tokens]
    boxes, classes, scores = boxes[order], classes[order], scores[order]
    normalized = boxes / np.asarray([width, height, width, height], dtype=np.float32)
    centers = (normalized[:, :2] + normalized[:, 2:]) / 2.0
    sizes = np.clip(normalized[:, 2:] - normalized[:, :2], 0.0, 1.0)
    area = sizes[:, 0] * sizes[:, 1]
    aspect = np.log(np.clip(sizes[:, 0] / np.maximum(sizes[:, 1], 1e-8), 1e-3, 1e3))
    values = np.column_stack((centers, sizes, scores, np.sqrt(area), aspect)).astype(np.float32)
    tokens = np.zeros((max_tokens, len(TOKEN_FEATURES)), dtype=np.float32)
    class_ids = np.zeros(max_tokens, dtype=np.int16)
    mask = np.zeros(max_tokens, dtype=bool)
    tokens[: len(values)] = values
    class_ids[: len(values)] = classes
    mask[: len(values)] = True
    return tokens, class_ids, mask


def sequence_id(image_name: str) -> str:
    """Return the VisDrone video prefix used for leakage-resistant splitting."""
    if "_" not in image_name:
        raise ValueError(f"VisDrone image name lacks a sequence prefix: {image_name}")
    return image_name.split("_", 1)[0]


def sequence_partition(sequence: str, fit_groups: set[str]) -> str:
    return "fit" if sequence in fit_groups else "holdout"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--fit-groups", type=int, default=160)
    args = parser.parse_args()
    predictions = read_jsonl(args.predictions)
    labels = {Path(item["image"]).name: item for item in read_jsonl(args.labels)}
    names = [Path(item["image"]).name for item in predictions]
    if len(names) != len(set(names)):
        raise ValueError("prediction image basenames must be unique")
    missing = [name for name in names if name not in labels]
    if missing:
        raise ValueError(f"utility label missing image: {missing[0]}")
    groups = sorted(
        {sequence_id(name) for name in names},
        key=lambda value: hashlib.sha256(f"p14:{value}".encode()).digest(),
    )
    if not 0 < args.fit_groups < len(groups):
        raise ValueError("fit_groups must leave at least one holdout sequence")
    fit_groups = set(groups[: args.fit_groups])
    encoded = [encode_detection_tokens(item, args.max_tokens, args.num_classes) for item in predictions]
    partitions = np.asarray([sequence_partition(sequence_id(name), fit_groups) for name in names])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        tokens=np.stack([item[0] for item in encoded]),
        classes=np.stack([item[1] for item in encoded]),
        mask=np.stack([item[2] for item in encoded]),
        names=np.asarray(names),
        sequences=np.asarray([sequence_id(name) for name in names]),
        partitions=partitions,
        utilities=np.asarray([float(labels[name]["utility"]) for name in names], dtype=np.float32),
        token_features=np.asarray(TOKEN_FEATURES),
    )
    metadata = {
        "status": "frozen scout detection cache; no val or test data",
        "predictions": str(args.predictions.resolve()),
        "predictions_sha256": sha256(args.predictions),
        "labels": str(args.labels.resolve()),
        "labels_sha256": sha256(args.labels),
        "images": len(names),
        "sequence_groups": len(groups),
        "fit_groups": args.fit_groups,
        "holdout_groups": len(groups) - args.fit_groups,
        "fit_images": int(np.sum(partitions == "fit")),
        "holdout_images": int(np.sum(partitions == "holdout")),
        "max_tokens": args.max_tokens,
        "token_features": list(TOKEN_FEATURES),
        "cache": str(args.output.resolve()),
        "cache_sha256": sha256(args.output),
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
