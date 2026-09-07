"""Create a deterministic, class-covered BDD100K smoke subset and its audit report."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


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
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def read_labels(path: Path) -> list[tuple[int, float, float, float, float]]:
    """Read one YOLO detection label file."""
    labels = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        values = line.split()
        if len(values) != 5:
            raise ValueError(f"{path}:{line_number}: expected 5 columns, got {len(values)}")
        class_id = int(values[0])
        if not 0 <= class_id < len(CLASS_NAMES):
            raise ValueError(f"{path}:{line_number}: invalid class ID {class_id}")
        coordinates = tuple(float(value) for value in values[1:])
        if not all(0.0 <= value <= 1.0 for value in coordinates):
            raise ValueError(f"{path}:{line_number}: coordinates outside [0, 1]")
        labels.append((class_id, *coordinates))
    return labels


def find_image(images_dir: Path, stem: str) -> Path:
    """Find the image corresponding to a label stem."""
    for suffix in IMAGE_SUFFIXES:
        candidate = images_dir / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No image found for {stem} under {images_dir}")


def select_split(root: Path, split: str, count: int, min_per_class: int, seed: int) -> list[Path]:
    """Select a deterministic subset with a minimum image quota for every available class."""
    labels_dir = root / "labels" / split
    images_dir = root / "images" / split
    label_paths = sorted(labels_dir.glob("*.txt"))
    if count > len(label_paths):
        raise ValueError(f"Requested {count} {split} images, but only {len(label_paths)} labels exist")

    rng = random.Random(seed)
    class_to_stems: dict[int, list[str]] = defaultdict(list)
    for label_path in label_paths:
        classes = {label[0] for label in read_labels(label_path)}
        for class_id in classes:
            class_to_stems[class_id].append(label_path.stem)

    selected: set[str] = set()
    for class_id in range(len(CLASS_NAMES)):
        candidates = class_to_stems[class_id][:]
        rng.shuffle(candidates)
        selected.update(candidates[:min_per_class])
    if len(selected) > count:
        raise ValueError(f"Class quotas selected {len(selected)} images, exceeding requested count {count}")

    remaining = [path.stem for path in label_paths if path.stem not in selected]
    rng.shuffle(remaining)
    selected.update(remaining[: count - len(selected)])
    return sorted(find_image(images_dir, stem) for stem in selected)


def write_list(root: Path, split: str, images: list[Path]) -> Path:
    """Write paths relative to the list file in Ultralytics-compatible form."""
    output = root / "splits" / f"smoke_{split}.txt"
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"./../images/{split}/{image.name}" for image in images]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def summarize(root: Path, split: str, images: list[Path]) -> dict:
    """Summarize classes, empty labels, and COCO-style object sizes for a selected split."""
    class_instances: Counter[int] = Counter()
    size_instances: Counter[str] = Counter()
    empty_labels = 0
    total_instances = 0
    for image_path in images:
        labels = read_labels(root / "labels" / split / f"{image_path.stem}.txt")
        if not labels:
            empty_labels += 1
        with Image.open(image_path) as image:
            image_width, image_height = image.size
        for class_id, _, _, width, height in labels:
            class_instances[class_id] += 1
            total_instances += 1
            area = width * image_width * height * image_height
            size_instances["small" if area < 32**2 else "medium" if area < 96**2 else "large"] += 1

    return {
        "images": len(images),
        "empty_labels": empty_labels,
        "instances": total_instances,
        "class_instances": {name: class_instances[index] for index, name in enumerate(CLASS_NAMES)},
        "size_instances_coco_area": {name: size_instances[name] for name in ("small", "medium", "large")},
        "small_instance_ratio": round(size_instances["small"] / total_instances, 6) if total_instances else 0.0,
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/BDD100K"))
    parser.add_argument("--train-count", type=int, default=400)
    parser.add_argument("--val-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    """Create split lists and a machine-readable audit report."""
    args = parse_args()
    root = args.root.resolve()
    selections = {
        "train": select_split(root, "train", args.train_count, min_per_class=5, seed=args.seed),
        "val": select_split(root, "val", args.val_count, min_per_class=2, seed=args.seed + 1),
    }
    outputs = {split: str(write_list(root, split, images)) for split, images in selections.items()}
    report = {
        "seed": args.seed,
        "selection": "class-covered deterministic sample; engineering smoke only",
        "lists": outputs,
        "splits": {split: summarize(root, split, images) for split, images in selections.items()},
    }
    report_path = root / "splits" / "smoke_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
