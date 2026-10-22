"""Build the BDD-10-class mapped view of VisDrone val for P32 zero-shot eval.

Reads native 10-class YOLO labels from data/VisDrone/labels/val, remaps class
indices per the frozen P32 mapping (pedestrian+people -> person, car -> car,
truck -> truck, bus -> bus, motor -> motor, bicycle -> bike; van / tricycle /
awning-tricycle dropped), and writes data/VisDrone/bdd10/{labels/val, images/val}.
Images are symlinked, never copied. Deterministic; safe to re-run (idempotent).
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

NATIVE_TO_BDD = {0: 0, 1: 0, 2: 7, 3: 2, 5: 3, 8: 4, 9: 6}
DROPPED = {4, 6, 7}
BDD_NAMES = {
    0: "person", 1: "rider", 2: "car", 3: "truck", 4: "bus",
    5: "train", 6: "motor", 7: "bike", 8: "traffic light", 9: "traffic sign",
}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    src_labels = root / "data/VisDrone/labels/val"
    dst_root = root / "data/VisDrone/bdd10"
    dst_labels = dst_root / "labels/val"
    dst_images = dst_root / "images/val"
    dst_labels.mkdir(parents=True, exist_ok=True)
    dst_images.parent.mkdir(parents=True, exist_ok=True)

    label_files = sorted(src_labels.glob("*.txt"))
    if not label_files:
        raise SystemExit(f"no source labels found under {src_labels}")
    expected_images = {p.stem for p in label_files}
    src_images = root / "data/VisDrone/images/val"
    actual_images = {p.stem for p in src_images.glob("*.jpg")}
    if expected_images != actual_images:
        raise SystemExit(
            f"label/image stem mismatch: labels={len(expected_images)} images={len(actual_images)}"
        )

    kept = Counter()
    dropped = Counter()
    for src in label_files:
        lines = []
        for raw in src.read_text(encoding="utf-8").splitlines():
            parts = raw.split()
            if not parts:
                continue
            native = int(float(parts[0]))
            if native in DROPPED:
                dropped[native] += 1
                continue
            mapped = NATIVE_TO_BDD[native]
            kept[mapped] += 1
            lines.append(" ".join([str(mapped)] + parts[1:]))
        (dst_labels / src.name).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    # Hardlink (never symlink): evaluate_object_sizes resolves prediction image
    # paths, and a symlinked images/val would resolve back to the native-label
    # tree. Hardlinks keep the bdd10 path stable under Path.resolve().
    if dst_images.is_symlink():
        dst_images.unlink()
    dst_images.mkdir(parents=True, exist_ok=True)
    import os

    linked = 0
    for src_img in sorted(src_images.glob("*.jpg")):
        dst_img = dst_images / src_img.name
        if dst_img.exists() and dst_img.samefile(src_img):
            continue
        if dst_img.exists():
            dst_img.unlink()
        os.link(src_img, dst_img)
        linked += 1
    print(f"images hardlinked: {linked} (total {len(list(dst_images.glob('*.jpg')))})")

    total_kept = sum(kept.values())
    total_dropped = sum(dropped.values())
    print(f"mapped {len(label_files)} label files -> {dst_labels}")
    print(f"instances kept: {total_kept}, dropped: {total_dropped}")
    for cls in sorted(kept):
        print(f"  bdd[{cls}] {BDD_NAMES[cls]}: {kept[cls]}")


if __name__ == "__main__":
    main()
