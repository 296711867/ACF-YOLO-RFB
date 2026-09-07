"""Verify that the frozen YOLO26 baseline can be imported and built."""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/yolo26-sdf-matplotlib")

import torch
from ultralytics import YOLO, __version__


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "upstream"


def main() -> None:
    """Print environment details and build the baseline and P2 variants."""
    print(f"ultralytics={__version__}")
    print(f"ultralytics_path={Path(__import__('ultralytics').__file__).resolve()}")
    print(f"torch={torch.__version__}")
    print(f"torch_cuda={torch.version.cuda}")
    print(f"cuda_available={torch.cuda.is_available()}")

    for model_name in ("yolo26.yaml", "yolo26-p2.yaml"):
        model_path = UPSTREAM / "ultralytics" / "cfg" / "models" / "26" / model_name
        model = YOLO(model_path, task="detect")
        detect = model.model.model[-1]
        print(
            f"{model_name}: layers={len(model.model.model)}, params={sum(p.numel() for p in model.model.parameters())}, "
            f"strides={detect.stride.tolist()}, end2end={detect.end2end}, reg_max={detect.reg_max}"
        )


if __name__ == "__main__":
    main()
