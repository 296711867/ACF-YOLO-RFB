"""Benchmark integrated 640-to-960 routing pipelines on one CUDA device."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import subprocess
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from ultralytics import YOLO

from scripts.evaluate_resolution_router import hash_random_scores, learned_scores
from scripts.run_sparse_refinement_feasibility import load_manifest
from scripts.train_perspective_router import top_fraction_indices
from sdf.refinement import PerspectiveBandRouter


POLICIES = ("full_640", "full_960", "random_25", "heuristic_25", "learned_25")


def sha256(path: Path) -> str:
    """Return a file SHA256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    """Return the implementation commit recorded by the benchmark."""
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True
    ).strip()


def policy_order(repeat: int) -> tuple[str, ...]:
    """Rotate policy order to reduce systematic thermal and cache bias."""
    offset = repeat % len(POLICIES)
    return POLICIES[offset:] + POLICIES[:offset]


def summarize_runs(runs: list[dict]) -> dict:
    """Summarize repeated timings and memory measurements."""
    keys = (
        "total_seconds",
        "coarse_seconds",
        "routing_seconds",
        "refinement_seconds",
        "throughput_images_per_second",
        "peak_allocated_mib",
        "peak_reserved_mib",
    )
    summary = {}
    for key in keys:
        values = [float(run[key]) for run in runs]
        summary[key] = {
            "mean": statistics.fmean(values),
            "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "values": values,
        }
    summary["routed_images"] = sorted({int(run["routed_images"]) for run in runs})
    summary["selection_sha256"] = sorted(
        {str(run["selection_sha256"]) for run in runs if run["selection_sha256"] is not None}
    )
    return summary


def selection_sha256(paths: list[Path], indices: np.ndarray) -> str:
    """Hash selected filename membership independent of score order."""
    names = "\n".join(sorted(paths[int(index)].name for index in indices)) + "\n"
    return hashlib.sha256(names.encode()).hexdigest()


def path_batches(paths: list[Path], batch: int) -> list[list[Path]]:
    """Split path lists explicitly because Ultralytics treats one list as one input batch."""
    if batch <= 0:
        raise ValueError("batch must be positive")
    return [paths[start : start + batch] for start in range(0, len(paths), batch)]


def load_router(checkpoint_path: Path, device: torch.device) -> PerspectiveBandRouter:
    """Load the frozen router head on the detector device."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    router = PerspectiveBandRouter(
        channels=int(checkpoint["channels"]),
        bands=int(checkpoint["bands"]),
        hidden=int(checkpoint["hidden"]),
    )
    router.load_state_dict(checkpoint["state_dict"])
    return router.eval().to(device)


def timed_cuda(call: Callable[[], object], device: torch.device) -> tuple[object, float]:
    """Time one operation with CUDA synchronization at both boundaries."""
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    value = call()
    torch.cuda.synchronize(device)
    return value, time.perf_counter() - started


def predict(detector: YOLO, paths: list[Path], imgsz: int, args: argparse.Namespace) -> list:
    """Run one batched Ultralytics prediction stage."""
    output = []
    for batch_paths in path_batches(paths, args.batch):
        options = dict(
            source=[str(path) for path in batch_paths],
            imgsz=imgsz,
            batch=args.batch,
            device=args.device,
            conf=args.conf,
            max_det=args.max_det,
            verbose=False,
            save=False,
            stream=True,
        )
        if args.half:
            options["half"] = True
        output.extend(result.cpu() for result in detector.predict(**options))
    return output


def heuristic_scores(results: list) -> np.ndarray:
    """Compute the frozen small-box confidence-sum policy from live results."""
    scores = []
    for result in results:
        boxes = result.boxes.xyxy.detach()
        confidence = result.boxes.conf.detach()
        areas = torch.clamp(boxes[:, 2:] - boxes[:, :2], min=0).prod(dim=1)
        scores.append(confidence[areas < 32**2].sum())
    return torch.stack(scores).float().cpu().numpy()


def run_policy(
    policy: str,
    detector: YOLO,
    router: PerspectiveBandRouter,
    paths: list[Path],
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    """Execute one complete policy and return synchronized phase timings."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    total_started = time.perf_counter()
    coarse_seconds = routing_seconds = refinement_seconds = 0.0
    routed_images = 0
    coarse_results = high_results = None
    selected = np.asarray([], dtype=np.int64)
    handle = None
    learned_score_parts: list[torch.Tensor] = []

    try:
        if policy == "full_960":
            high_results, refinement_seconds = timed_cuda(lambda: predict(detector, paths, 960, args), device)
        else:
            if policy == "learned_25":

                def capture(_module, _inputs, output) -> None:
                    with torch.inference_mode():
                        learned_score_parts.append(router(output.detach().float()))

                handle = detector.model.model[args.router_layer].register_forward_hook(capture)
            coarse_results, coarse_seconds = timed_cuda(lambda: predict(detector, paths, 640, args), device)
            if handle is not None:
                handle.remove()
                handle = None

            if policy != "full_640":

                def select() -> np.ndarray:
                    if policy == "random_25":
                        scores = hash_random_scores([str(path) for path in paths], args.random_seed)
                    elif policy == "heuristic_25":
                        scores = heuristic_scores(coarse_results)
                    else:
                        scores = torch.cat(learned_score_parts).detach().cpu().numpy()
                        if len(scores) != len(paths):
                            raise RuntimeError(f"router produced {len(scores)} scores for {len(paths)} images")
                    return top_fraction_indices(scores, args.route_fraction)

                selected, routing_seconds = timed_cuda(select, device)
                routed_images = len(selected)
                routed_paths = [paths[int(index)] for index in selected]
                high_results, refinement_seconds = timed_cuda(
                    lambda: predict(detector, routed_paths, 960, args), device
                )
                combined_results = list(coarse_results)
                for routed_result, index in zip(high_results, selected):
                    combined_results[int(index)] = routed_result
                high_results = combined_results
        torch.cuda.synchronize(device)
        total_seconds = time.perf_counter() - total_started
        return {
            "total_seconds": total_seconds,
            "coarse_seconds": coarse_seconds,
            "routing_seconds": routing_seconds,
            "refinement_seconds": refinement_seconds,
            "throughput_images_per_second": len(paths) / total_seconds,
            "routed_images": routed_images,
            "selection_sha256": selection_sha256(paths, selected) if routed_images else None,
            "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
        }
    finally:
        if handle is not None:
            handle.remove()
        del coarse_results, high_results, learned_score_parts


def parse_args() -> argparse.Namespace:
    """Parse the frozen runtime protocol."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--router-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-learned-scores", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--images", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-images", type=int, default=16)
    parser.add_argument("--route-fraction", type=float, default=0.25)
    parser.add_argument("--router-layer", type=int, default=19)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--half", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace, available_images: int) -> None:
    """Reject protocol settings that cannot produce a meaningful comparison."""
    if args.images <= 0 or args.images > available_images:
        raise ValueError(f"images must be in [1, {available_images}]")
    if args.repeats < 2:
        raise ValueError("at least two repeats are required for sample standard deviation")
    if args.batch <= 0 or args.warmup_images <= 0:
        raise ValueError("batch and warmup-images must be positive")
    if not 0 < args.route_fraction <= 1:
        raise ValueError("route-fraction must be in (0, 1]")


def main() -> None:
    """Warm up one model, execute rotated policies, and write raw and aggregate evidence."""
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the integrated runtime benchmark")
    all_paths = load_manifest(args.manifest, args.images_dir)
    validate_args(args, len(all_paths))
    paths = all_paths[: args.images]
    device = torch.device(f"cuda:{args.device}" if str(args.device).isdigit() else args.device)
    detector = YOLO(args.model)
    router = load_router(args.router_checkpoint, device)

    warmup_paths = paths[: min(args.warmup_images, len(paths))]
    for policy in POLICIES:
        run_policy(policy, detector, router, warmup_paths, args, device)
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()

    runs = {policy: [] for policy in POLICIES}
    execution_order = []
    for repeat in range(args.repeats):
        order = policy_order(repeat)
        execution_order.append(list(order))
        for policy in order:
            result = run_policy(policy, detector, router, paths, args, device)
            result["repeat"] = repeat
            runs[policy].append(result)
            print(json.dumps({"policy": policy, **result}), flush=True)

    summary = {policy: summarize_runs(values) for policy, values in runs.items()}
    expected_scores = learned_scores(args.expected_learned_scores, [str(path) for path in paths])
    expected_selection = selection_sha256(paths, top_fraction_indices(expected_scores, args.route_fraction))
    if summary["learned_25"]["selection_sha256"] != [expected_selection]:
        raise RuntimeError(
            "integrated learned route does not match the frozen offline Top-fraction selection: "
            f"{summary['learned_25']['selection_sha256']} != {expected_selection}"
        )
    full_640 = summary["full_640"]["total_seconds"]["mean"]
    full_960 = summary["full_960"]["total_seconds"]["mean"]
    for policy, values in summary.items():
        elapsed = values["total_seconds"]["mean"]
        values["relative_total_percent"] = {
            "vs_full_640": (elapsed / full_640 - 1) * 100,
            "vs_full_960": (elapsed / full_960 - 1) * 100,
        }

    effective_manifest = "\n".join(path.name for path in paths) + "\n"
    report = {
        "status": "frozen integrated runtime benchmark; performance evidence only",
        "protocol": {
            "implementation_commit": git_commit(),
            "model": str(args.model.resolve()),
            "model_sha256": sha256(args.model),
            "router_checkpoint": str(args.router_checkpoint.resolve()),
            "router_checkpoint_sha256": sha256(args.router_checkpoint),
            "expected_learned_scores": str(args.expected_learned_scores.resolve()),
            "expected_learned_scores_sha256": sha256(args.expected_learned_scores),
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": sha256(args.manifest),
            "effective_manifest_sha256": hashlib.sha256(effective_manifest.encode()).hexdigest(),
            "images_dir": str(args.images_dir.resolve()),
            "images": len(paths),
            "batch": args.batch,
            "repeats": args.repeats,
            "warmup_images_per_policy": len(warmup_paths),
            "route_fraction": args.route_fraction,
            "routed_images": math.ceil(len(paths) * args.route_fraction),
            "confidence": args.conf,
            "max_det": args.max_det,
            "half": args.half,
            "execution_order": execution_order,
            "expected_learned_selection_sha256": expected_selection,
            "timing_boundary": (
                "image paths to a complete materialized CPU Ultralytics Results list, including routed replacement; "
                "synchronized CUDA; model load and per-policy warmup excluded"
            ),
        },
        "environment": {
            "hostname": platform.node(),
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device),
            "gpu_total_mib": torch.cuda.get_device_properties(device).total_memory / 2**20,
        },
        "runs": runs,
        "summary": summary,
        "interpretation_limit": (
            "This benchmark measures one server, one batch setting, and a fixed 500-image prefix. "
            "It does not change the frozen accuracy decision and is not a cross-hardware deployment claim."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
