"""Run one real-weight CUDA/AMP backward pass for a frozen P2-to-P3 config."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from ultralytics.cfg import get_cfg
from ultralytics.nn.distill_model import DistillationModel
from ultralytics.nn.tasks import DetectionModel, load_checkpoint
from ultralytics.utils import YAML


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    """Return the SHA256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def synthetic_dense_batch(batch_size: int, imgsz: int, device: torch.device) -> dict[str, torch.Tensor]:
    """Build a deterministic dense batch with 48 mixed-size targets per image."""
    generator = torch.Generator(device=device).manual_seed(0)
    targets_per_image = 48
    count = batch_size * targets_per_image
    centers = 0.05 + 0.9 * torch.rand(count, 2, generator=generator, device=device)
    sizes = 0.015 + 0.185 * torch.rand(count, 2, generator=generator, device=device)
    return {
        "img": torch.rand(batch_size, 3, imgsz, imgsz, generator=generator, device=device),
        "batch_idx": torch.arange(batch_size, device=device).repeat_interleave(targets_per_image).float(),
        "cls": (torch.arange(count, device=device) % 10).float().view(-1, 1),
        "bboxes": torch.cat((centers, sizes), dim=1),
    }


def main() -> None:
    """Load one frozen config and report loss, gradients, time, and peak CUDA memory."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Stage 20 batch preflight")

    config_path = args.config.resolve()
    config = YAML.load(config_path)
    device = torch.device("cuda:0")
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    model_path = (ROOT / config["model"]).resolve()
    pretrained_path = (ROOT / config["pretrained"]).resolve()
    teacher_path = (ROOT / config["distill_model"]).resolve() if config.get("distill_model") else None
    for path in (model_path, pretrained_path, teacher_path):
        if path is not None and not path.is_file():
            raise FileNotFoundError(path)

    student = DetectionModel(model_path, ch=3, nc=10, verbose=False).to(device)
    pretrained = load_checkpoint(pretrained_path, device=device)[0]
    student.load(pretrained)
    overrides = {key: value for key, value in config.items() if key != "stage_stop_epoch"}
    student.args = get_cfg(overrides=overrides)
    model = DistillationModel(teacher_path, student) if teacher_path else student
    model.train()
    batch = synthetic_dense_batch(config["batch"], config["imgsz"], device)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.float16):
        if isinstance(model, DistillationModel):
            loss, loss_items = model.loss(batch)
        else:
            predictions = model(batch["img"])
            loss, loss_items = model.loss(batch, predictions)
    loss.sum().backward()
    torch.cuda.synchronize(device)

    trainable_gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    teacher_gradients = (
        [] if teacher_path is None else [parameter.grad for parameter in model.teacher_model.parameters()]
    )
    report = {
        "status": "passed",
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "mode": config.get("distill_mode", "d0_control"),
        "batch": config["batch"],
        "imgsz": config["imgsz"],
        "targets": len(batch["bboxes"]),
        "loss": [float(value) for value in loss.detach().cpu()],
        "loss_items": {key: float(value) for key, value in loss_items.items()},
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated(device) / 2**30, 3),
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved(device) / 2**30, 3),
        "finite_trainable_gradients": all(
            gradient is not None and torch.isfinite(gradient).all() for gradient in trainable_gradients
        ),
        "teacher_has_gradient": any(gradient is not None for gradient in teacher_gradients),
        "teacher_sha256": sha256(teacher_path) if teacher_path else None,
    }
    if not all(torch.isfinite(value).all() for value in (loss, *loss_items.values())):
        raise RuntimeError(f"Non-finite loss in preflight: {report}")
    if not report["finite_trainable_gradients"] or report["teacher_has_gradient"]:
        raise RuntimeError(f"Invalid gradients in preflight: {report}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
