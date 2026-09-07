"""Safely resume an interrupted formal experiment in its original run directory."""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.utils import YAML

try:
    from .run_experiment import (
        ROOT,
        StageBoundaryReached,
        git,
        sha256,
        stage_stop_callback,
        validate_devices,
        validate_stage_checkpoint,
    )
except ImportError:  # Direct script execution.
    from run_experiment import (
        ROOT,
        StageBoundaryReached,
        git,
        sha256,
        stage_stop_callback,
        validate_devices,
        validate_stage_checkpoint,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--stop-epoch", type=int, help="Pause after saving this epoch with optimizer state intact")
    parser.add_argument("--preflight", action="store_true", help="Validate resume state without launching training")
    return parser.parse_args()


def append_history(path: Path, event: dict) -> None:
    """Append one event to the resume history JSON file."""
    history = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []
    history.append(event)
    path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")


def update_output_metadata(metadata: dict, run_dir: Path, best_sha256: str, last_sha256: str) -> None:
    """Record the current final checkpoint paths and hashes in experiment metadata."""
    metadata["outputs"] = {
        "best": {"path": str(run_dir / "weights" / "best.pt"), "sha256": best_sha256},
        "last": {"path": str(run_dir / "weights" / "last.pt"), "sha256": last_sha256},
    }


def resume_weight_check_callback(checkpoint: dict):
    """Return a callback that requires the live model to exactly match checkpoint EMA before training."""
    expected = checkpoint["ema"].float().state_dict()

    def verify_weights(trainer) -> None:
        actual = trainer.model.state_dict()
        mismatched = [
            key
            for key, value in expected.items()
            if key not in actual or not torch.equal(actual[key].detach().cpu(), value.detach().cpu())
        ]
        if mismatched:
            raise RuntimeError(f"Resume model does not match checkpoint EMA; mismatched tensors: {mismatched[:5]}")
        print(f"Verified exact resume weights: {len(expected)} tensors match checkpoint EMA")

    return verify_weights


def main() -> None:
    """Validate a resumable checkpoint and continue training in place."""
    args = parse_args()
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/yolo26-sdf-matplotlib")
    run_dir = args.run_dir.resolve()
    runs_root = (ROOT / "runs").resolve()
    if not run_dir.is_relative_to(runs_root):
        raise ValueError(f"Run directory must be under {runs_root}: {run_dir}")

    dirty = git("status", "--porcelain")
    if dirty:
        raise RuntimeError("Refusing formal resume from a dirty project tree:\n" + dirty)

    metadata_path = run_dir / "experiment_metadata.json"
    config_path = run_dir / "frozen_config.yaml"
    results_path = run_dir / "results.csv"
    checkpoint_path = run_dir / "weights" / "last.pt"
    required_paths = (metadata_path, config_path, results_path, checkpoint_path)
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete run, missing: {missing}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    config = YAML.load(config_path)
    original_config_path = Path(metadata["files"]["config"]["path"])
    expected_config_hash = metadata["files"]["config"]["sha256"]
    if sha256(config_path) != expected_config_hash or sha256(original_config_path) != expected_config_hash:
        raise RuntimeError("Frozen or current training config no longer matches the original config hash")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    completed_epochs = int(checkpoint.get("epoch", -1)) + 1
    planned_epochs = int(checkpoint.get("train_args", {}).get("epochs", 0))
    if completed_epochs <= 0 or completed_epochs >= planned_epochs:
        raise RuntimeError(f"Checkpoint is not incomplete: completed={completed_epochs}, planned={planned_epochs}")
    if args.stop_epoch is not None and not completed_epochs < args.stop_epoch < planned_epochs:
        raise ValueError(
            f"stop_epoch must be between completed and planned epochs: "
            f"completed={completed_epochs}, stop={args.stop_epoch}, planned={planned_epochs}"
        )
    for key in ("optimizer", "scaler", "ema"):
        if checkpoint.get(key) is None:
            raise RuntimeError(f"Checkpoint cannot resume exactly because '{key}' is missing")
    rows = list(csv.DictReader(results_path.open(encoding="utf-8")))
    if len(rows) != completed_epochs or int(rows[-1]["epoch"]) != completed_epochs:
        raise RuntimeError(f"CSV/checkpoint epoch mismatch: rows={len(rows)}, checkpoint={completed_epochs}")

    device = config["device"]
    cuda_devices = validate_devices(device)
    lock_path = run_dir / "resume.lock"
    if lock_path.exists():
        raise FileExistsError(f"Resume lock already exists; inspect the recorded PID before removal: {lock_path}")

    evidence = {
        "status": "preflight_passed" if args.preflight else "running",
        "timestamp": datetime.now().astimezone().isoformat(),
        "resume_commit": git("rev-parse", "HEAD"),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "completed_epochs": completed_epochs,
        "next_epoch": completed_epochs + 1,
        "planned_epochs": planned_epochs,
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "visible_cuda_devices": cuda_devices,
        "pid": os.getpid(),
    }
    if args.preflight:
        print(json.dumps(evidence, indent=2))
        return

    lock_path.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    history_path = run_dir / "resume_history.json"
    append_history(history_path, evidence)
    model = YOLO(checkpoint_path)
    model.add_callback("on_train_start", resume_weight_check_callback(checkpoint))
    if args.stop_epoch:
        model.add_callback("on_model_save", stage_stop_callback(args.stop_epoch))
    try:
        model.train(resume=True, device=device)
    except StageBoundaryReached as boundary:
        validate_stage_checkpoint(checkpoint_path, args.stop_epoch)
        evidence["status"] = "paused_at_stage"
        evidence["ended_at"] = datetime.now().astimezone().isoformat()
        evidence["stop_epoch"] = args.stop_epoch
        evidence["reason"] = str(boundary)
        evidence["best_sha256"] = sha256(run_dir / "weights" / "best.pt")
        evidence["last_sha256"] = sha256(checkpoint_path)
        append_history(history_path, evidence)
        metadata["status"] = "paused_at_stage"
        metadata["ended_at"] = evidence["ended_at"]
        metadata["resume_commit"] = evidence["resume_commit"]
        metadata["stage"] = {"stop_epoch": args.stop_epoch, "reason": str(boundary)}
        update_output_metadata(metadata, run_dir, evidence["best_sha256"], evidence["last_sha256"])
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(f"Paused resumed run at resumable epoch {args.stop_epoch}: {run_dir}")
    except BaseException as error:
        evidence["status"] = "failed"
        evidence["ended_at"] = datetime.now().astimezone().isoformat()
        evidence["error"] = repr(error)
        append_history(history_path, evidence)
        raise
    else:
        evidence["status"] = "completed"
        evidence["ended_at"] = datetime.now().astimezone().isoformat()
        evidence["best_sha256"] = sha256(run_dir / "weights" / "best.pt")
        evidence["last_sha256"] = sha256(run_dir / "weights" / "last.pt")
        append_history(history_path, evidence)
        metadata["status"] = "completed_after_resume"
        metadata["ended_at"] = evidence["ended_at"]
        metadata["resume_commit"] = evidence["resume_commit"]
        update_output_metadata(metadata, run_dir, evidence["best_sha256"], evidence["last_sha256"])
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    finally:
        lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
