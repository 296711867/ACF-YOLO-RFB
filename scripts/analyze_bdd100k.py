"""Generate full BDD100K class, object-size, resolution, and scene statistics."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import lance


CLASS_NAMES = ("person", "rider", "car", "truck", "bus", "train", "motor", "bike", "traffic light", "traffic sign")
SPLITS = ("train", "val")


def empty_stats() -> dict:
    """Create mutable counters for one split."""
    return {
        "images": 0,
        "instances": 0,
        "empty_images": 0,
        "class_instances": Counter(),
        "class_images": Counter(),
        "sizes": Counter(),
        "weather": Counter(),
        "scene": Counter(),
        "timeofday": Counter(),
        "resolutions": Counter(),
        "occluded": 0,
        "truncated": 0,
        "invalid_boxes_skipped": 0,
        "duplicate_boxes_skipped": 0,
    }


def finalize(stats: dict) -> dict:
    """Convert counters to ordered JSON-compatible dictionaries."""
    instances = stats["instances"]
    return {
        "images": stats["images"],
        "instances": instances,
        "empty_images": stats["empty_images"],
        "instances_per_image": round(instances / stats["images"], 6),
        "class_instances": {name: stats["class_instances"][name] for name in CLASS_NAMES},
        "class_images": {name: stats["class_images"][name] for name in CLASS_NAMES},
        "size_instances_coco_area": {name: stats["sizes"][name] for name in ("small", "medium", "large")},
        "small_instance_ratio": round(stats["sizes"]["small"] / instances, 6),
        "weather": dict(stats["weather"].most_common()),
        "scene": dict(stats["scene"].most_common()),
        "timeofday": dict(stats["timeofday"].most_common()),
        "resolutions": dict(stats["resolutions"].most_common()),
        "occluded_instances": stats["occluded"],
        "truncated_instances": stats["truncated"],
        "invalid_boxes_skipped": stats["invalid_boxes_skipped"],
        "duplicate_boxes_skipped": stats["duplicate_boxes_skipped"],
    }


def analyze(lance_path: Path) -> dict:
    """Scan the verified Lance snapshot while excluding mirror-only synthetic rows."""
    columns = [
        "image_id", "split", "width", "height", "weather", "scene", "timeofday", "ann_categories", "ann_bboxes",
        "ann_occluded", "ann_truncated",
    ]
    stats = {split: empty_stats() for split in SPLITS}
    dataset = lance.dataset(lance_path)
    for batch in dataset.to_batches(columns=columns, batch_size=2048):
        for row in batch.to_pylist():
            split = row["split"]
            if split not in stats or row["image_id"].startswith("synthetic_"):
                continue
            target = stats[split]
            target["images"] += 1
            target["weather"][row["weather"] or "undefined"] += 1
            target["scene"][row["scene"] or "undefined"] += 1
            target["timeofday"][row["timeofday"] or "undefined"] += 1
            target["resolutions"][f"{row['width']}x{row['height']}"] += 1
            categories = row["ann_categories"] or []
            boxes = row["ann_bboxes"] or []
            occluded = row["ann_occluded"] or []
            truncated = row["ann_truncated"] or []
            if not categories:
                target["empty_images"] += 1
            present_classes = set()
            seen_boxes = set()
            for index, (category, box) in enumerate(zip(categories, boxes)):
                if category not in CLASS_NAMES or len(box) != 4:
                    target["invalid_boxes_skipped"] += 1
                    continue
                x1, y1, x2, y2 = (float(value) for value in box)
                box_width = max(0.0, min(x2, row["width"]) - max(0.0, x1))
                box_height = max(0.0, min(y2, row["height"]) - max(0.0, y1))
                if box_width <= 0 or box_height <= 0:
                    target["invalid_boxes_skipped"] += 1
                    continue
                signature = (
                    category,
                    round((max(0.0, x1) + min(x2, row["width"])) / (2 * row["width"]), 6),
                    round((max(0.0, y1) + min(y2, row["height"])) / (2 * row["height"]), 6),
                    round(box_width / row["width"], 6),
                    round(box_height / row["height"], 6),
                )
                if signature in seen_boxes:
                    target["duplicate_boxes_skipped"] += 1
                    continue
                seen_boxes.add(signature)
                area = box_width * box_height
                target["sizes"]["small" if area < 32**2 else "medium" if area < 96**2 else "large"] += 1
                target["instances"] += 1
                target["class_instances"][category] += 1
                present_classes.add(category)
                target["occluded"] += int(index < len(occluded) and occluded[index])
                target["truncated"] += int(index < len(truncated) and truncated[index])
            target["class_images"].update(present_classes)
    return {
        "source": str(lance_path.resolve()),
        "synthetic_rows_excluded": True,
        "size_protocol": "COCO pixel area: small < 32^2, medium < 96^2, large >= 96^2",
        "splits": {split: finalize(stats[split]) for split in SPLITS},
    }


def markdown(report: dict) -> str:
    """Render a concise human-readable summary."""
    lines = [
        "# BDD100K 全量基础统计",
        "",
        "> 生成日期：2026-07-25（Asia/Shanghai）  ",
        "> 来源：已审计 Lance 标注快照；排除 500 条 `synthetic_val_*`。",
        "",
        "## 汇总",
        "",
        "| split | images | instances | empty | instances/image | small | medium | large | small ratio |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split, stats in report["splits"].items():
        sizes = stats["size_instances_coco_area"]
        lines.append(
            f"| {split} | {stats['images']:,} | {stats['instances']:,} | {stats['empty_images']:,} | "
            f"{stats['instances_per_image']:.3f} | {sizes['small']:,} | {sizes['medium']:,} | {sizes['large']:,} | "
            f"{stats['small_instance_ratio']:.2%} |"
        )
    lines.extend(["", "## 分类别实例", "", "| class | train | val |", "|---|---:|---:|"])
    for name in CLASS_NAMES:
        lines.append(
            f"| {name} | {report['splits']['train']['class_instances'][name]:,} | "
            f"{report['splits']['val']['class_instances'][name]:,} |"
        )
    for key, title in (("weather", "天气"), ("scene", "场景"), ("timeofday", "时段")):
        lines.extend(["", f"## {title}", "", "| value | train | val |", "|---|---:|---:|"])
        values = list(dict.fromkeys([*report["splits"]["train"][key], *report["splits"]["val"][key]]))
        for value in values:
            lines.append(
                f"| {value} | {report['splits']['train'][key].get(value, 0):,} | "
                f"{report['splits']['val'][key].get(value, 0):,} |"
            )
    lines.extend(["", "完整分辨率、遮挡、截断和类别图像数见同目录 `stats.json`。", ""])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--lance-path", type=Path, default=Path("data/BDD100K/label_source/data/bdd100k.lance"))
    parser.add_argument("--output", type=Path, default=Path("results/dataset_stats/bdd100k"))
    return parser.parse_args()


def main() -> None:
    """Generate JSON and Markdown statistics."""
    args = parse_args()
    report = analyze(args.lance_path.resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "stats.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output / "README.md").write_text(markdown(report), encoding="utf-8")
    print(json.dumps(report["splits"], indent=2))


if __name__ == "__main__":
    main()
