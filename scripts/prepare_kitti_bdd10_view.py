"""Build the BDD-10-class mapped view of KITTI object trainval for P33 zero-shot eval.

Reads native KITTI labels from data/KITTI/raw/training/label_2 (KITTI format:
type truncated occlusion alpha l t r b ...), keeps Car / Pedestrian / Cyclist
per the frozen P33 mapping (Car -> car, Pedestrian -> person, Cyclist -> bike;
Van / Person_sitting / Tram / Misc / DontCare dropped), converts absolute-pixel
bboxes to normalized YOLO boxes, and writes data/KITTI/bdd10/{labels/val,
images/val}. Images are hardlinked (never symlinked: evaluate_object_sizes
resolves prediction image paths; a symlink would resolve back to the native
label tree). Deterministic; idempotent. Image sizes are read per file (KITTI
images have non-uniform resolutions).
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

KEPT = {"Car": 2, "Pedestrian": 0, "Cyclist": 7}
DROPPED = {"Van", "Person_sitting", "Tram", "Misc", "DontCare"}
BDD_NAMES = {
    0: "person", 1: "rider", 2: "car", 3: "truck", 4: "bus",
    5: "train", 6: "motor", 7: "bike", 8: "traffic light", 9: "traffic sign",
}


def main() -> None:
    from PIL import Image

    root = Path(__file__).resolve().parents[1]
    src_labels = root / "data/KITTI/raw/training/label_2"
    src_images = root / "data/KITTI/raw/training/image_2"
    dst_root = root / "data/KITTI/bdd10"
    dst_labels = dst_root / "labels/val"
    dst_images = dst_root / "images/val"
    dst_labels.mkdir(parents=True, exist_ok=True)
    dst_images.parent.mkdir(parents=True, exist_ok=True)

    label_files = sorted(src_labels.glob("*.txt"))
    if not label_files:
        raise SystemExit(f"no source labels found under {src_labels}")
    expected = {p.stem for p in label_files}
    actual = {p.stem for p in src_images.glob("*.png")}
    if expected != actual:
        missing = sorted(expected - actual)[:5]
        extra = sorted(actual - expected)[:5]
        raise SystemExit(f"label/image stem mismatch: labels-only={missing} images-only={extra}")

    kept: Counter = Counter()
    dropped: Counter = Counter()
    for src in label_files:
        img_path = src_images / (src.stem + ".png")
        w, h = Image.open(img_path).size
        lines = []
        for raw in src.read_text(encoding="utf-8").splitlines():
            parts = raw.split()
            if len(parts) < 5:
                continue
            kind = parts[0]
            if kind in DROPPED:
                dropped[kind] += 1
                continue
            mapped = KEPT.get(kind)
            if mapped is None:
                dropped[kind] += 1
                continue
            l, t, r, b = (float(parts[4 + i]) for i in range(4))
            bw, bh = r - l, b - t
            if bw <= 0 or bh <= 0:
                dropped[kind] += 1
                continue
            cx, cy = l + bw / 2, t + bh / 2
            kept[mapped] += 1
            lines.append(" ".join([
                str(mapped), f"{cx / w:.6f}", f"{cy / h:.6f}", f"{bw / w:.6f}", f"{bh / h:.6f}",
            ]))
        (dst_labels / (src.stem + ".txt")).write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    import os

    dst_images.mkdir(parents=True, exist_ok=True)
    linked = 0
    for src_img in sorted(src_images.glob("*.png")):
        dst_img = dst_images / src_img.name
        if dst_img.exists() and dst_img.samefile(src_img):
            continue
        if dst_img.exists():
            dst_img.unlink()
        os.link(src_img, dst_img)
        linked += 1
    print(f"images hardlinked: {linked} (total {len(list(dst_images.glob('*.png')))})")

    total_kept = sum(kept.values())
    total_dropped = sum(dropped.values())
    print(f"mapped {len(label_files)} label files -> {dst_labels}")
    print(f"instances kept: {total_kept}, dropped: {total_dropped}")
    for cls in sorted(kept):
        print(f"  bdd[{cls}] {BDD_NAMES[cls]}: {kept[cls]}")


if __name__ == "__main__":
    main()
