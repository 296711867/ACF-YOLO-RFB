"""Build detector-free RGB thumbnail caches with frozen utility labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from scripts.run_sparse_refinement_feasibility import load_manifest


STATISTIC_NAMES = (
    "gray_mean",
    "gray_std",
    "laplacian_abs_mean",
    "sobel_magnitude_mean",
    "band0_gray_std",
    "band0_laplacian_abs_mean",
    "band1_gray_std",
    "band1_laplacian_abs_mean",
    "band2_gray_std",
    "band2_laplacian_abs_mean",
    "band3_gray_std",
    "band3_laplacian_abs_mean",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_utility_labels(path: Path) -> dict[str, dict]:
    """Index labels by basename and reject ambiguous duplicate image names."""
    labels = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        name = Path(item["image"]).name
        if name in labels:
            raise ValueError(f"duplicate utility label basename: {name}")
        labels[name] = item
    return labels


def thumbnail_statistics(rgb: np.ndarray) -> np.ndarray:
    """Compute frozen low-cost image-statistic baselines from one RGB thumbnail."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("thumbnail must have HWC RGB shape")
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    values = [
        float(gray.mean()),
        float(gray.std()),
        float(np.abs(laplacian).mean()),
        float(np.hypot(sobel_x, sobel_y).mean()),
    ]
    for band in np.array_split(gray, 4, axis=0):
        values.extend((float(band.std()), float(np.abs(cv2.Laplacian(band, cv2.CV_32F)).mean())))
    return np.asarray(values, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.width != 160 or args.height != 96:
        raise ValueError("the frozen thumbnail size is 160x96")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive")

    paths = load_manifest(args.manifest, args.images_dir)
    if args.limit is not None:
        paths = paths[: args.limit]
    labels = load_utility_labels(args.labels)
    missing = [path.name for path in paths if path.name not in labels]
    if missing:
        raise ValueError(f"utility labels missing image: {missing[0]}")

    thumbnails = np.empty((len(paths), args.height, args.width, 3), dtype=np.uint8)
    statistics = np.empty((len(paths), len(STATISTIC_NAMES)), dtype=np.float32)
    utilities = np.empty(len(paths), dtype=np.float32)
    ranks = np.empty(len(paths), dtype=np.int32)
    for index, path in enumerate(paths):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"failed to decode image: {path}")
        resized = cv2.resize(image, (args.width, args.height), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        thumbnails[index] = rgb
        statistics[index] = thumbnail_statistics(rgb)
        utilities[index] = float(labels[path.name]["utility"])
        ranks[index] = int(labels[path.name]["rank"])
        if (index + 1) % 500 == 0 or index + 1 == len(paths):
            print(f"thumbnails: {index + 1}/{len(paths)}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        images=thumbnails,
        names=np.asarray([path.name for path in paths]),
        utilities=utilities,
        ranks=ranks,
        statistics=statistics,
        statistic_names=np.asarray(STATISTIC_NAMES),
    )
    metadata = {
        "status": "detector-free thumbnail cache; not formal accuracy evidence",
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "labels": str(args.labels.resolve()),
        "labels_sha256": sha256(args.labels),
        "images_dir": str(args.images_dir.resolve()),
        "images": len(paths),
        "thumbnail_shape_nhwc": list(thumbnails.shape),
        "thumbnail_dtype": str(thumbnails.dtype),
        "statistic_names": list(STATISTIC_NAMES),
        "utility_min": float(utilities.min()),
        "utility_max": float(utilities.max()),
        "utility_mean": float(utilities.mean()),
        "cache": str(args.output.resolve()),
        "cache_sha256": sha256(args.output),
    }
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
