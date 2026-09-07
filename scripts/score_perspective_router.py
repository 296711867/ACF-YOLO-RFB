"""Score frozen P2 descriptors with a trained perspective-band router."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from sdf.refinement import PerspectiveBandRouter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    router = PerspectiveBandRouter(
        channels=int(checkpoint["channels"]),
        bands=int(checkpoint["bands"]),
        hidden=int(checkpoint["hidden"]),
    )
    router.load_state_dict(checkpoint["state_dict"])
    router.eval()
    with np.load(args.features) as data:
        images = data["images"]
        descriptors = torch.from_numpy(data["descriptors"]).float()
    with torch.no_grad():
        scores = router.forward_descriptor(descriptors).numpy()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output_file:
        for image, score in zip(images, scores):
            output_file.write(json.dumps({"image": str(image), "score": float(score)}) + "\n")
    print(
        json.dumps(
            {
                "images": len(images),
                "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                "features_sha256": hashlib.sha256(args.features.read_bytes()).hexdigest(),
                "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
