"""Extract compact perspective-band descriptors from a frozen P2 detector."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

from scripts.run_sparse_refinement_feasibility import load_manifest
from sdf.refinement import perspective_band_descriptor


def load_labels(path: Path | None) -> dict[str, dict]:
    """Index optional routing labels by resolved image path."""
    if path is None:
        return {}
    labels = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        labels[str(Path(item["image"]).resolve())] = item
    return labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--layer", type=int, default=19)
    parser.add_argument("--bands", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    paths = load_manifest(args.manifest, args.images_dir)
    detector = YOLO(args.model)
    captured: list[torch.Tensor] = []

    def capture(_module, _inputs, output) -> None:
        captured.append(perspective_band_descriptor(output.detach().float(), args.bands).cpu())

    handle = detector.model.model[args.layer].register_forward_hook(capture)
    try:
        for start in range(0, len(paths), args.batch):
            batch_paths = paths[start : start + args.batch]
            detector.predict(
                source=[str(path) for path in batch_paths],
                imgsz=args.imgsz,
                device=args.device,
                conf=0.001,
                max_det=300,
                verbose=False,
                save=False,
            )
            if (start + len(batch_paths)) % 500 == 0 or start + len(batch_paths) == len(paths):
                print(f"features: {start + len(batch_paths)}/{len(paths)}", flush=True)
    finally:
        handle.remove()
    descriptors = torch.cat(captured).numpy()
    if len(descriptors) < len(paths):
        raise RuntimeError(f"captured {len(descriptors)} descriptors for {len(paths)} images")
    descriptors = descriptors[-len(paths) :]

    labels = load_labels(args.labels)
    utilities = np.asarray([labels.get(str(path), {}).get("utility", np.nan) for path in paths], dtype=np.float32)
    ranks = np.asarray([labels.get(str(path), {}).get("rank", -1) for path in paths], dtype=np.int32)
    if labels and np.isnan(utilities).any():
        raise ValueError("labels do not cover every manifest image")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        images=np.asarray([path.name for path in paths]),
        descriptors=descriptors,
        utilities=utilities,
        ranks=ranks,
    )
    metadata = {
        "model": str(args.model.resolve()),
        "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "labels": str(args.labels.resolve()) if args.labels else None,
        "images": len(paths),
        "layer": args.layer,
        "bands": args.bands,
        "descriptor_shape": list(descriptors.shape),
        "imgsz": args.imgsz,
    }
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
