"""Build the BDD-10-class mapped view of RTTS for P34 zero-shot eval.

Reads VOC XML annotations from data/RTTS/annotations_xml, maps classes per the
frozen P34 mapping (person -> person, car -> car, bus -> bus, motorbike ->
motor, bicycle -> bike; no drops needed - the annotation inventory contains
exactly these five classes), converts absolute-pixel VOC boxes to normalized
YOLO boxes (image size read per file via PIL; RTTS images have 2000+ distinct
resolutions), and writes data/RTTS/bdd10/{labels/val, images/val}. Images are
hardlinked (never symlinked: evaluate_object_sizes resolves prediction image
paths). Deterministic; idempotent. All 4322 images are used as the eval set;
the bundled ImageSets folder is intentionally not consulted (no split
selection - same discipline as P33).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

KEPT = {"person": 0, "car": 2, "bus": 4, "motorbike": 6, "bicycle": 7}
BDD_NAMES = {
    0: "person", 1: "rider", 2: "car", 3: "truck", 4: "bus",
    5: "train", 6: "motor", 7: "bike", 8: "traffic light", 9: "traffic sign",
}


def main() -> None:
    from PIL import Image

    root = Path(__file__).resolve().parents[1]
    src_xml = root / "data/RTTS/annotations_xml"
    src_images = root / "data/RTTS/JPEGImages"
    dst_root = root / "data/RTTS/bdd10"
    dst_labels = dst_root / "labels/val"
    dst_images = dst_root / "images/val"
    dst_labels.mkdir(parents=True, exist_ok=True)
    dst_images.mkdir(parents=True, exist_ok=True)

    xml_files = sorted(src_xml.glob("*.xml"))
    if not xml_files:
        raise SystemExit(f"no source annotations found under {src_xml}")
    expected = {p.stem for p in xml_files}
    actual = {p.stem for p in src_images.glob("*.png")}
    if expected != actual:
        missing = sorted(expected - actual)[:5]
        extra = sorted(actual - expected)[:5]
        raise SystemExit(f"xml/image stem mismatch: xml-only={missing} images-only={extra}")

    kept: Counter = Counter()
    skipped: Counter = Counter()
    for src in xml_files:
        img_path = src_images / (src.stem + ".png")
        w, h = Image.open(img_path).size
        lines = []
        tree = ET.parse(src)
        for obj in tree.findall("object"):
            name = (obj.find("name").text or "").strip()
            mapped = KEPT.get(name)
            if mapped is None:
                skipped[name] += 1
                continue
            bb = obj.find("bndbox")
            l, t = float(bb.find("xmin").text), float(bb.find("ymin").text)
            r, b = float(bb.find("xmax").text), float(bb.find("ymax").text)
            bw, bh = r - l, b - t
            if bw <= 0 or bh <= 0:
                skipped[name] += 1
                continue
            cx, cy = l + bw / 2, t + bh / 2
            kept[mapped] += 1
            lines.append(" ".join([
                str(mapped), f"{cx / w:.6f}", f"{cy / h:.6f}", f"{bw / w:.6f}", f"{bh / h:.6f}",
            ]))
        (dst_labels / (src.stem + ".txt")).write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    import os

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
    print(f"mapped {len(xml_files)} label files -> {dst_labels}")
    print(f"instances kept: {total_kept}, skipped: {sum(skipped.values())}")
    for cls in sorted(kept):
        print(f"  bdd[{cls}] {BDD_NAMES[cls]}: {kept[cls]}")
    for name, n in skipped.most_common():
        print(f"  skipped[{name}]: {n}")


if __name__ == "__main__":
    main()
