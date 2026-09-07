"""Download VisDrone2019-DET from the Ultralytics mirror and convert it to YOLO format."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from PIL import Image

from ultralytics.utils import ASSETS_URL, TQDM
from ultralytics.utils.downloads import download


SPLITS = {
    "train": ("VisDrone2019-DET-train", 6471),
    "val": ("VisDrone2019-DET-val", 548),
    "test": ("VisDrone2019-DET-test-dev", 1610),
}


def deduplicate_labels(labels_dir: Path) -> int:
    """Remove exact duplicate YOLO rows while preserving their original order."""
    removed = 0
    for label_path in labels_dir.glob("*.txt"):
        lines = label_path.read_text(encoding="utf-8").splitlines()
        unique_lines = list(dict.fromkeys(lines))
        removed += len(lines) - len(unique_lines)
        if len(unique_lines) != len(lines):
            text = "\n".join(unique_lines)
            label_path.write_text(f"{text}\n" if text else "", encoding="utf-8")
    return removed


def convert_split(root: Path, split: str, source_name: str) -> None:
    """Convert one original VisDrone split into Ultralytics YOLO layout."""
    source_dir = root / source_name
    source_images = source_dir / "images"
    source_annotations = source_dir / "annotations"
    images_dir = root / "images" / split
    labels_dir = root / "labels" / split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    for image_path in source_images.glob("*.jpg"):
        image_path.replace(images_dir / image_path.name)

    annotations = sorted(source_annotations.glob("*.txt"))
    for annotation_path in TQDM(annotations, desc=f"Converting VisDrone {split}"):
        image_path = images_dir / annotation_path.with_suffix(".jpg").name
        width, height = Image.open(image_path).size
        lines = []
        for row in (line.split(",") for line in annotation_path.read_text(encoding="utf-8").splitlines()):
            if len(row) < 6 or row[4] == "0":
                continue
            x, y, box_width, box_height = map(int, row[:4])
            class_id = int(row[5]) - 1
            x_center = (x + box_width / 2) / width
            y_center = (y + box_height / 2) / height
            lines.append(
                f"{class_id} {x_center:.6f} {y_center:.6f} "
                f"{box_width / width:.6f} {box_height / height:.6f}\n"
            )
        (labels_dir / annotation_path.name).write_text("".join(lines), encoding="utf-8")

    shutil.rmtree(source_dir)


def prepare(root: Path) -> None:
    """Download, convert, and verify all public VisDrone2019-DET splits."""
    root.mkdir(parents=True, exist_ok=True)
    for split, (source_name, expected_images) in SPLITS.items():
        images_dir = root / "images" / split
        labels_dir = root / "labels" / split
        image_count = len(list(images_dir.glob("*.jpg")))
        label_count = len(list(labels_dir.glob("*.txt")))
        if image_count == label_count == expected_images:
            duplicates = deduplicate_labels(labels_dir)
            print(f"VisDrone {split}: already complete ({image_count} images, {duplicates} duplicate boxes removed)")
            continue

        url = f"{ASSETS_URL}/{source_name}.zip"
        print(f"Downloading {url}")
        download([url], dir=root, threads=1)
        convert_split(root, split, source_name)
        duplicates = deduplicate_labels(labels_dir)

        image_count = len(list(images_dir.glob("*.jpg")))
        label_count = len(list(labels_dir.glob("*.txt")))
        if image_count != expected_images or label_count != expected_images:
            raise RuntimeError(
                f"VisDrone {split} verification failed: images={image_count}, labels={label_count}, "
                f"expected={expected_images}"
            )
        print(f"VisDrone {split}: verified ({image_count} images and labels, {duplicates} duplicate boxes removed)")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/VisDrone"))
    return parser.parse_args()


if __name__ == "__main__":
    prepare(parse_args().root.resolve())
