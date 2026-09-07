"""Render deterministic label-audit images and contact sheets for a YOLO dataset."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


COLORS = (
    "#ff3b30",
    "#34c759",
    "#007aff",
    "#ff9500",
    "#af52de",
    "#5ac8fa",
    "#ff2d55",
    "#a2845e",
    "#ffd60a",
    "#64d2ff",
)


def load_names(data_yaml: Path) -> list[str]:
    """Load class names from the small YAML subset used by this project."""
    names = {}
    in_names = False
    for line in data_yaml.read_text(encoding="utf-8").splitlines():
        if line.strip() == "names:":
            in_names = True
            continue
        if in_names and line and not line.startswith((" ", "\t")):
            break
        if in_names and ":" in line:
            key, value = line.strip().split(":", 1)
            if key.isdigit():
                names[int(key)] = value.strip().strip("'\"")
    if not names or sorted(names) != list(range(len(names))):
        raise ValueError(f"Could not parse contiguous class names from {data_yaml}")
    return [names[index] for index in range(len(names))]


def label_path(image_path: Path) -> Path:
    """Map an images path to its YOLO labels path."""
    parts = list(image_path.parts)
    index = len(parts) - 1 - parts[::-1].index("images")
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def render(image_path: Path, names: list[str], output: Path) -> dict:
    """Draw YOLO boxes on one image and return audit metadata."""
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    boxes = 0
    for line_number, line in enumerate(label_path(image_path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        values = line.split()
        if len(values) != 5:
            raise ValueError(f"{label_path(image_path)}:{line_number}: expected 5 columns")
        class_id = int(values[0])
        x_center, y_center, box_width, box_height = (float(value) for value in values[1:])
        x1 = (x_center - box_width / 2) * width
        y1 = (y_center - box_height / 2) * height
        x2 = (x_center + box_width / 2) * width
        y2 = (y_center + box_height / 2) * height
        color = COLORS[class_id % len(COLORS)]
        line_width = max(2, round(min(width, height) / 300))
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)
        text = names[class_id]
        text_box = draw.textbbox((x1, y1), text, font=ImageFont.load_default())
        draw.rectangle(text_box, fill=color)
        draw.text((x1, y1), text, fill="black", font=ImageFont.load_default())
        boxes += 1
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, quality=88, optimize=True)
    return {"source": str(image_path), "rendered": str(output), "boxes": boxes}


def contact_sheets(rendered: list[Path], output_dir: Path, per_sheet: int = 20) -> list[str]:
    """Create compact contact sheets for efficient manual inspection."""
    outputs = []
    thumb_width, thumb_height = 320, 200
    columns, rows = 4, 5
    for sheet_index, start in enumerate(range(0, len(rendered), per_sheet), 1):
        sheet = Image.new("RGB", (columns * thumb_width, rows * thumb_height), "#202124")
        draw = ImageDraw.Draw(sheet)
        for index, path in enumerate(rendered[start : start + per_sheet]):
            with Image.open(path) as source:
                image = source.convert("RGB")
                image.thumbnail((thumb_width, thumb_height - 18))
            x = (index % columns) * thumb_width
            y = (index // columns) * thumb_height
            sheet.paste(image, (x + (thumb_width - image.width) // 2, y))
            draw.text((x + 4, y + thumb_height - 16), path.stem, fill="white")
        output = output_dir / f"contact_sheet_{sheet_index:03d}.jpg"
        sheet.save(output, quality=90, optimize=True)
        outputs.append(str(output))
    return outputs


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    """Render a fixed random audit sample and write a manifest."""
    args = parse_args()
    images = sorted(path for path in args.images.resolve().glob("*.*") if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if args.count > len(images):
        raise ValueError(f"Requested {args.count} images, found {len(images)}")
    selected = random.Random(args.seed).sample(images, args.count)
    output_dir = args.output.resolve()
    rendered_dir = output_dir / "images"
    names = load_names(args.data)
    records = [render(path, names, rendered_dir / f"{path.stem}.jpg") for path in selected]
    sheets = contact_sheets([Path(record["rendered"]) for record in records], output_dir)
    manifest = {"seed": args.seed, "count": len(records), "records": records, "contact_sheets": sheets}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"count": len(records), "contact_sheets": sheets}, indent=2))


if __name__ == "__main__":
    main()
