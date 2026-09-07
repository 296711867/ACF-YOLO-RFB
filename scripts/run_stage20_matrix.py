"""Run a frozen Stage 20 experiment matrix sequentially."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from ultralytics.utils import YAML


ROOT = Path(__file__).resolve().parents[1]
MATRICES = {
    "p2p3": {
        "directory": ROOT / "runs/detect/phase4-distill-stage20",
        "configs": (
            ROOT / "configs/train/bdd100k-p2p3-d0-control-640-e200-stop20-b16.yaml",
            ROOT / "configs/train/bdd100k-p2p3-d1-common-640-e200-stop20-b16.yaml",
            ROOT / "configs/train/bdd100k-p2p3-d2-small-640-e200-stop20-b16.yaml",
            ROOT / "configs/train/bdd100k-p2p3-d3-consensus-640-e200-stop20-b16.yaml",
        ),
    },
    "rdci": {
        "directory": ROOT / "runs/detect/phase5-rdci-stage20",
        "configs": (
            ROOT / "configs/train/bdd100k-rdci-r0-p2-control-640-e200-stop20-b32.yaml",
            ROOT / "configs/train/bdd100k-rdci-r1-candidate-640-e200-stop20-b32.yaml",
        ),
    },
    "pascp": {
        "directory": ROOT / "runs/detect/phase6-pascp-stage20",
        "configs": (
            ROOT / "configs/train/bdd100k-pascp-r0-p2-control-640-e200-stop20-b32.yaml",
            ROOT / "configs/train/bdd100k-pascp-r1-candidate-640-e200-stop20-b32.yaml",
        ),
    },
}


def now() -> str:
    """Return an aware local timestamp."""
    return datetime.now().astimezone().isoformat()


def write_status(status: dict, status_path: Path) -> None:
    """Atomically replace the matrix status file."""
    temporary = status_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    temporary.replace(status_path)


def run_logged(command: list[str], environment: dict[str, str], log_path: Path) -> None:
    """Stream one child process to both the terminal and persistent matrix log."""
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{now()}] command: {' '.join(command)}\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def parse_args() -> argparse.Namespace:
    """Parse the frozen matrix name."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", choices=tuple(MATRICES), required=True)
    return parser.parse_args()


def main() -> None:
    """Run every group only after its predecessor reaches a valid resumable Stage 20 boundary."""
    selection = MATRICES[parse_args().matrix]
    matrix_dir = selection["directory"]
    status_path = matrix_dir / "matrix_status.json"
    lock_path = matrix_dir / "matrix.lock"
    log_path = matrix_dir / "matrix.log"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    lock = lock_path.open("x", encoding="utf-8")
    lock.write(json.dumps({"pid": os.getpid(), "started_at": now()}) + "\n")
    lock.close()
    status = {"status": "running", "started_at": now(), "groups": []}
    write_status(status, status_path)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "0"
    environment.setdefault("MPLCONFIGDIR", "/tmp/yolo26-sdf-matplotlib")
    environment.setdefault("PYTHONPYCACHEPREFIX", "/tmp/yolo26-pycache-p2p3-stage20")
    environment["PYTHONPATH"] = f"{ROOT / 'upstream'}:{ROOT}"

    try:
        for config_path in selection["configs"]:
            config = YAML.load(config_path)
            run_dir = (ROOT / config["project"] / config["name"]).resolve()
            if run_dir.exists():
                raise FileExistsError(f"Refusing existing matrix run: {run_dir}")
            group = {"config": str(config_path), "name": config["name"], "status": "running", "started_at": now()}
            status["groups"].append(group)
            write_status(status, status_path)
            run_logged(
                [sys.executable, str(ROOT / "scripts/run_experiment.py"), "--config", str(config_path)],
                environment,
                log_path,
            )
            metadata_path = run_dir / "experiment_metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("status") != "paused_at_stage" or metadata.get("stage", {}).get("stop_epoch") != 20:
                raise RuntimeError(f"Invalid Stage 20 completion for {config_path}: {metadata.get('status')}")
            group.update(
                {
                    "status": "paused_at_stage",
                    "ended_at": now(),
                    "git_commit": metadata["git"]["commit"],
                    "outputs": metadata.get("outputs", {}),
                }
            )
            write_status(status, status_path)
        status["status"] = "completed"
        status["ended_at"] = now()
        write_status(status, status_path)
    except BaseException as error:
        status["status"] = "failed"
        status["ended_at"] = now()
        status["error"] = repr(error)
        write_status(status, status_path)
        raise
    finally:
        lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
