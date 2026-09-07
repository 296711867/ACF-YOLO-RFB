"""Create a deterministic BDD training-image manifest for refinement screening."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def ranked_names(images_dir: Path, seed: int) -> list[str]:
    """Return image names in a stable hash-defined order."""
    names = [path.name for path in images_dir.glob("*.jpg")]
    return sorted(names, key=lambda name: (hashlib.sha256(f"{seed}:{name}".encode()).digest(), name))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=260807)
    args = parser.parse_args()
    if args.count <= 0:
        raise ValueError("count must be positive")
    if args.offset < 0:
        raise ValueError("offset must be non-negative")
    names = ranked_names(args.images_dir, args.seed)
    if len(names) < args.offset + args.count:
        raise ValueError(f"requested range ends at {args.offset + args.count}, found {len(names)} images")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(names[args.offset : args.offset + args.count]) + "\n", encoding="utf-8")
    print(f"wrote {args.count} names at offset {args.offset} to {args.output}")


if __name__ == "__main__":
    main()
