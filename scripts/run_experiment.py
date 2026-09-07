"""Launch a frozen experiment with Git, config, weight, and environment evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch
from ultralytics import YOLO, __version__ as ultralytics_version
from ultralytics.utils import YAML


ROOT = Path(__file__).resolve().parents[1]
PROJECT_GIT_DIR = ROOT / ".project.git"


class StageBoundaryReached(RuntimeError):
    """Signal that a resumable checkpoint was saved at a configured stage boundary."""


def sha256(path: Path) -> str:
    """Return the SHA256 of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    """Run Git against the alternate repository or the current standard worktree."""
    command = (
        ["git", f"--git-dir={PROJECT_GIT_DIR}", f"--work-tree={ROOT}", *args]
        if PROJECT_GIT_DIR.is_dir()
        else ["git", "-C", str(ROOT), *args]
    )
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true", help="Validate provenance and paths without starting a run")
    return parser.parse_args()


def validate_devices(device: str | int) -> list[str]:
    """Validate requested CUDA indices and return all visible device names."""
    visible_devices = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    device_text = str(device).strip().lower()
    if device_text in {"cpu", "mps"}:
        return visible_devices
    requested = [int(index.strip()) for index in device_text.split(",")]
    missing = [index for index in requested if index < 0 or index >= len(visible_devices)]
    if missing:
        raise RuntimeError(
            f"Requested CUDA devices {requested}, but only {len(visible_devices)} are visible: {visible_devices}"
        )
    return visible_devices


def stage_stop_callback(stop_epoch: int):
    """Return a callback that pauses immediately after saving the requested epoch."""

    def stop_after_save(trainer) -> None:
        if trainer.epoch + 1 >= stop_epoch:
            raise StageBoundaryReached(f"Saved resumable stage checkpoint at epoch {stop_epoch}")

    return stop_after_save


def validate_stage_checkpoint(path: Path, stop_epoch: int) -> None:
    """Require a checkpoint with complete state at the requested stage boundary."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    completed_epochs = int(checkpoint.get("epoch", -1)) + 1
    if completed_epochs != stop_epoch:
        raise RuntimeError(f"Stage checkpoint epoch mismatch: expected={stop_epoch}, actual={completed_epochs}")
    missing = [key for key in ("optimizer", "scaler", "ema") if checkpoint.get(key) is None]
    if missing:
        raise RuntimeError(f"Stage checkpoint is not resumable; missing state: {missing}")


def main() -> None:
    """Validate provenance, reserve the run directory, and launch training."""
    args = parse_args()
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/yolo26-sdf-matplotlib")
    config_path = args.config.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    dirty = git("status", "--porcelain")
    if dirty:
        raise RuntimeError("Refusing formal experiment from a dirty project tree:\n" + dirty)

    config = YAML.load(config_path)
    required = {"task", "model", "data", "project", "name", "exist_ok", "seed", "device"}
    missing = required - config.keys()
    if missing:
        raise ValueError(f"Missing frozen config fields: {sorted(missing)}")
    if config["exist_ok"]:
        raise ValueError("Formal config must set exist_ok: false")
    stage_stop_epoch = config.pop("stage_stop_epoch", None)
    if stage_stop_epoch is not None and not 0 < int(stage_stop_epoch) < int(config["epochs"]):
        raise ValueError("stage_stop_epoch must be positive and lower than epochs")
    stage_stop_epoch = int(stage_stop_epoch) if stage_stop_epoch is not None else None
    cuda_devices = validate_devices(config["device"])

    model_path = (ROOT / config.pop("model")).resolve()
    data_path = (ROOT / config["data"]).resolve()
    pretrained_path = None
    if isinstance(config.get("pretrained"), str):
        pretrained_path = (ROOT / config["pretrained"]).resolve()
        config["pretrained"] = str(pretrained_path)
    teacher_path = None
    if isinstance(config.get("distill_model"), str):
        teacher_path = (ROOT / config["distill_model"]).resolve()
        config["distill_model"] = str(teacher_path)
    project_path = (ROOT / config["project"]).resolve()
    run_dir = project_path / config["name"]
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {run_dir}")
    if (
        not model_path.is_file()
        or not data_path.is_file()
        or (pretrained_path and not pretrained_path.is_file())
        or (teacher_path and not teacher_path.is_file())
    ):
        raise FileNotFoundError(
            f"Missing model, data config, or pretrained weights: "
            f"model={model_path}, data={data_path}, pretrained={pretrained_path}, teacher={teacher_path}"
        )

    commit = git("rev-parse", "HEAD")
    metadata = {
        "status": "starting",
        "started_at": datetime.now().astimezone().isoformat(),
        "command": [sys.executable, str(Path(__file__).resolve()), "--config", str(config_path)],
        "git": {"directory": git("rev-parse", "--absolute-git-dir"), "commit": commit, "clean": True},
        "files": {
            "config": {"path": str(config_path), "sha256": sha256(config_path)},
            "data": {"path": str(data_path), "sha256": sha256(data_path)},
            "model": {"path": str(model_path), "sha256": sha256(model_path)},
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "ultralytics": ultralytics_version,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cuda_devices": cuda_devices,
        },
    }
    if pretrained_path:
        metadata["files"]["pretrained"] = {"path": str(pretrained_path), "sha256": sha256(pretrained_path)}
    if teacher_path:
        metadata["files"]["teacher"] = {"path": str(teacher_path), "sha256": sha256(teacher_path)}
    if args.preflight:
        print(
            json.dumps(
                {
                    "status": "preflight_passed",
                    "run_dir": str(run_dir),
                    "git_commit": commit,
                    "config_sha256": metadata["files"]["config"]["sha256"],
                    "data_sha256": metadata["files"]["data"]["sha256"],
                    "model_sha256": metadata["files"]["model"]["sha256"],
                    "pretrained_sha256": metadata["files"].get("pretrained", {}).get("sha256"),
                    "teacher_sha256": metadata["files"].get("teacher", {}).get("sha256"),
                    "cuda_devices": cuda_devices,
                },
                indent=2,
            )
        )
        return
    run_dir.mkdir(parents=True)
    metadata_path = run_dir / "experiment_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    shutil.copy2(config_path, run_dir / "frozen_config.yaml")

    task = config.pop("task")
    config.pop("mode", None)
    config["model"] = str(model_path)
    config["data"] = str(data_path)
    config["project"] = str(project_path)
    config["exist_ok"] = True  # Safe because the wrapper exclusively reserved a previously absent run directory.
    model = YOLO(model_path, task=task)
    if stage_stop_epoch:
        model.add_callback("on_model_save", stage_stop_callback(stage_stop_epoch))
    try:
        model.train(**config)
    except StageBoundaryReached as boundary:
        last = run_dir / "weights" / "last.pt"
        validate_stage_checkpoint(last, stage_stop_epoch)
        metadata["status"] = "paused_at_stage"
        metadata["ended_at"] = datetime.now().astimezone().isoformat()
        metadata["stage"] = {"stop_epoch": stage_stop_epoch, "reason": str(boundary)}
        metadata["outputs"] = {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in (("best", run_dir / "weights" / "best.pt"), ("last", last))
            if path.is_file()
        }
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(f"Paused formal run at resumable epoch {stage_stop_epoch}: {run_dir}")
        return
    except BaseException as error:
        metadata["status"] = "failed"
        metadata["ended_at"] = datetime.now().astimezone().isoformat()
        metadata["error"] = repr(error)
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        raise

    actual_run = Path(model.trainer.save_dir).resolve()
    if actual_run != run_dir:
        raise RuntimeError(f"Unexpected save directory: expected {run_dir}, got {actual_run}")
    metadata["status"] = "completed"
    metadata["ended_at"] = datetime.now().astimezone().isoformat()
    metadata["outputs"] = {
        name: {"path": str(path), "sha256": sha256(path)}
        for name, path in (("best", run_dir / "weights" / "best.pt"), ("last", run_dir / "weights" / "last.pt"))
        if path.is_file()
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Completed formal run at {run_dir}")


if __name__ == "__main__":
    main()
