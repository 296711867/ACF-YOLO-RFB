"""Convert verified BDD100K train/val annotations from a local Lance snapshot to YOLO format."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import lance


CLASS_NAMES = (
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motor",
    "bike",
    "traffic light",
    "traffic sign",
)
CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}
EXPECTED_IMAGES = {"train": 70000, "val": 10000}


def ensure_image_links(root: Path) -> None:
    """Expose the official 100k split under images/ so Ultralytics can map images to labels."""
    images_dir = root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    for split in EXPECTED_IMAGES:
        link = images_dir / split
        if link.is_dir() and not link.is_symlink():
            continue
        target = Path("..") / "100k" / split
        if link.is_symlink() and link.readlink() == target:
            continue
        if link.exists() or link.is_symlink():
            raise FileExistsError(f"Refusing to replace existing path: {link}")
        link.symlink_to(target, target_is_directory=True)


def convert(root: Path, lance_path: Path) -> dict:
    """Convert official train and val rows, excluding mirror-specific synthetic rows."""
    ensure_image_links(root)
    dataset = lance.dataset(lance_path)
    columns = ["image_id", "split", "width", "height", "ann_categories", "ann_bboxes"]
    table = dataset.to_table(columns=columns)

    split_counts = Counter()
    class_counts = Counter()
    skipped_synthetic = 0
    invalid_boxes = 0
    clipped_boxes = 0
    duplicate_boxes = 0
    missing_images = []

    for batch in table.to_batches(max_chunksize=2048):
        for row in batch.to_pylist():
            image_id = row["image_id"]
            split = row["split"]
            if image_id.startswith("synthetic_"):
                skipped_synthetic += 1
                continue
            if split not in EXPECTED_IMAGES:
                continue

            image_path = root / "images" / split / f"{image_id}.jpg"
            if not image_path.is_file():
                missing_images.append(str(image_path))
                continue

            width, height = row["width"], row["height"]
            lines = []
            seen_lines = set()
            for category, box in zip(row["ann_categories"], row["ann_bboxes"]):
                if category not in CLASS_TO_ID or len(box) != 4:
                    invalid_boxes += 1
                    continue
                x1, y1, x2, y2 = map(float, box)
                clipped = (
                    max(0.0, min(x1, width)),
                    max(0.0, min(y1, height)),
                    max(0.0, min(x2, width)),
                    max(0.0, min(y2, height)),
                )
                if clipped != (x1, y1, x2, y2):
                    clipped_boxes += 1
                x1, y1, x2, y2 = clipped
                box_width, box_height = x2 - x1, y2 - y1
                if box_width <= 0 or box_height <= 0:
                    invalid_boxes += 1
                    continue
                x_center = (x1 + x2) / (2 * width)
                y_center = (y1 + y2) / (2 * height)
                line = (
                    f"{CLASS_TO_ID[category]} {x_center:.6f} {y_center:.6f} "
                    f"{box_width / width:.6f} {box_height / height:.6f}\n"
                )
                if line in seen_lines:
                    duplicate_boxes += 1
                    continue
                seen_lines.add(line)
                lines.append(line)
                class_counts[category] += 1

            labels_dir = root / "labels" / split
            labels_dir.mkdir(parents=True, exist_ok=True)
            (labels_dir / f"{image_id}.txt").write_text("".join(lines), encoding="utf-8")
            split_counts[split] += 1

    if missing_images:
        raise FileNotFoundError(f"Missing {len(missing_images)} images; first entries: {missing_images[:10]}")
    if dict(split_counts) != EXPECTED_IMAGES:
        raise RuntimeError(f"Unexpected converted split counts: {dict(split_counts)}, expected {EXPECTED_IMAGES}")

    report = {
        "source_rows": dataset.count_rows(),
        "converted_images": dict(split_counts),
        "skipped_synthetic_rows": skipped_synthetic,
        "class_names": list(CLASS_NAMES),
        "class_instances": {name: class_counts[name] for name in CLASS_NAMES},
        "invalid_boxes_skipped": invalid_boxes,
        "boxes_clipped_to_image": clipped_boxes,
        "duplicate_boxes_skipped": duplicate_boxes,
    }
    (root / "conversion_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/BDD100K"))
    parser.add_argument(
        "--lance-path",
        type=Path,
        default=Path("data/BDD100K/label_source/data/bdd100k.lance"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(convert(args.root.resolve(), args.lance_path.resolve()), indent=2))
