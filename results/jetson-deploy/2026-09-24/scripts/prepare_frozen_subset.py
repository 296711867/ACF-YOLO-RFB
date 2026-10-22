"""[RTX SERVER SIDE] Prepare the frozen evaluation subset package for Jetson.

Run this on the training server (it needs the BDD100K val split), NOT on the
Jetson. It produces a single tar.gz holding:

  images/           N val images chosen by the frozen md5-order rule
  labels/           matching YOLO-format ground truth
  bench_100/        the 100 frozen speed-test images (§11.3 list)
  subset_manifest.json, scene_manifest.json, subset_gt_stats.json

Transfer the tar.gz next to this package on the Jetson and unpack it into
jetson_deploy/data/frozen_subset/.

  python3 scripts/prepare_frozen_subset.py \
      --val-images /path/to/data/BDD100K/images/val \
      --val-labels /path/to/data/BDD100K/labels/val \
      --num 1000 --make-tar
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-images", type=Path, required=True)
    parser.add_argument("--val-labels", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=PACKAGE_ROOT / "data" / "frozen_subset")
    parser.add_argument("--num", type=int, default=1000)
    parser.add_argument("--bench-list", type=Path, default=PACKAGE_ROOT / "data" / "bench_images_100.txt")
    parser.add_argument("--make-tar", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    from coco_eval_core import load_ground_truth_label
    from subset_lib import frozen_order, image_brightness, scene_tags, select_scenes

    args = parse_args()
    all_names = sorted(p.name for p in args.val_images.glob("*.jpg"))
    if not all_names:
        raise SystemExit(f"no jpgs under {args.val_images}")
    bench_names = [n for n in args.bench_list.read_text().splitlines() if n.strip()]
    missing_bench = [n for n in bench_names if n not in set(all_names)]
    if missing_bench:
        raise SystemExit(f"bench list images missing from val split: {missing_bench[:5]}")

    subset = [n for n in frozen_order(all_names) if n not in set(bench_names)][: args.num]
    bench_set = set(bench_names)
    chosen = subset + bench_names  # bench 100 travels with the subset, kept as its own dir

    img_out = args.out / "images"
    lab_out = args.out / "labels"
    bench_out = args.out / "bench_100"
    for d in (img_out, lab_out, bench_out):
        d.mkdir(parents=True, exist_ok=True)

    print(f"[copy] {len(chosen)} images -> {args.out}")
    files_meta, scene_records, class_instances = [], [], {}
    import cv2

    for index, name in enumerate(chosen):
        src_img = args.val_images / name
        src_lab = args.val_labels / (Path(name).stem + ".txt")
        if name in bench_set:
            shutil.copy2(src_img, bench_out / name)
            files_meta.append({"image": name, "sha256": sha256(src_img), "role": "bench_100"})
            continue
        shutil.copy2(src_img, img_out / name)
        label_target = lab_out / (Path(name).stem + ".txt")
        if src_lab.is_file():
            shutil.copyfile(src_lab, label_target)  # byte-exact copy, no CRLF translation on Windows
        else:
            label_target.write_bytes(b"")

        img = cv2.imread(str(src_img))
        h, w = img.shape[:2]
        gt_boxes, gt_classes, _ = load_ground_truth_label(src_lab if src_lab.is_file() else label_target, w, h)
        brightness = image_brightness(img)
        tags = scene_tags(gt_boxes, brightness)
        score = {
            "day": brightness,
            "night": -brightness,
            "dense": float(len(gt_boxes)),
            "distant": float(np.mean(np.sqrt((gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1]))) < 24)
            if len(gt_boxes)
            else 0.0,
            "occlusion": float(len(gt_boxes)),
        }
        scene_records.append({"image": name, "tags": tags, "score_by_tag": score, "brightness": round(brightness, 1)})
        for cls in gt_classes.tolist():
            class_instances[int(cls)] = class_instances.get(int(cls), 0) + 1
        files_meta.append({"image": name, "sha256": sha256(src_img), "role": "subset"})
        if (index + 1) % 200 == 0:
            print(f"  {index + 1}/{len(chosen)}")

    manifest = {
        "rule": f"md5(filename) ascending, first {args.num} after excluding the §11.3 bench 100; bench 100 appended as bench_100/",
        "val_images_root": str(args.val_images),
        "num_subset": len(subset),
        "num_bench": len(bench_names),
        "files": files_meta,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (args.out / "subset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    scene_manifest = select_scenes(scene_records)
    (args.out / "scene_manifest.json").write_text(json.dumps(scene_manifest, indent=2) + "\n", encoding="utf-8")

    names = {int(k): v for k, v in json.loads((PACKAGE_ROOT / "data" / "bdd10k_classes.json").read_text()).items()}
    (args.out / "subset_gt_stats.json").write_text(
        json.dumps(
            {
                "images": len(subset),
                "instances_per_class": {names[k]: v for k, v in sorted(class_instances.items())},
                "total_instances": sum(class_instances.values()),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"[done] subset at {args.out}: {len(subset)} eval + {len(bench_names)} bench images")
    if args.make_tar:
        tar_path = args.out.with_suffix(".tar.gz")
        subprocess.run(["tar", "czf", str(tar_path), "-C", str(args.out), "."], check=True)
        print(f"[tar] {tar_path} ({tar_path.stat().st_size / 1e6:.1f} MB)")
        print(f"next: scp {tar_path} <jetson>:{PACKAGE_ROOT.name}/data/ && cd data && tar xzf {tar_path.name} -C frozen_subset")


if __name__ == "__main__":
    main()
