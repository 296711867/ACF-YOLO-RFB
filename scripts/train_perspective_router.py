"""Train the tiny perspective-band router on frozen P2 descriptors."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from sdf.refinement import PerspectiveBandRouter


def top_fraction_indices(scores: np.ndarray, fraction: float) -> np.ndarray:
    """Return stable descending-score indices for an exact fixed budget."""
    count = math.ceil(len(scores) * fraction)
    return np.asarray(sorted(range(len(scores)), key=lambda index: (-float(scores[index]), index))[:count])


def routing_metrics(scores: np.ndarray, utilities: np.ndarray, targets: np.ndarray, fraction: float) -> dict:
    """Measure top-budget target precision and positive-utility capture."""
    selected = top_fraction_indices(scores, fraction)
    oracle = top_fraction_indices(utilities, fraction)
    denominator = float(utilities[oracle].sum())
    return {
        "selected": len(selected),
        "target_precision": float(targets[selected].mean()),
        "utility_sum": float(utilities[selected].sum()),
        "oracle_utility_sum": denominator,
        "utility_capture": float(utilities[selected].sum() / denominator) if denominator else 0.0,
    }


def load_features(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-features", type=Path, required=True)
    parser.add_argument("--dev-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--route-fraction", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train = load_features(args.train_features)
    dev = load_features(args.dev_features)
    descriptors = torch.from_numpy(train["descriptors"]).float()
    utilities = train["utilities"].astype(np.float32)
    target_cutoff = math.ceil(len(utilities) * args.route_fraction)
    targets = (train["ranks"] <= target_cutoff).astype(np.float32)
    fit_count = int(len(descriptors) * 0.8)
    fit_indices = torch.arange(fit_count)
    holdout_indices = torch.arange(fit_count, len(descriptors))

    router = PerspectiveBandRouter(channels=32, bands=4, hidden=32)
    optimizer = torch.optim.AdamW(router.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    fit_targets = torch.from_numpy(targets[:fit_count])
    positive = float(fit_targets.sum())
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((fit_count - positive) / positive))
    generator = torch.Generator().manual_seed(args.seed)
    best, stale = None, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        router.train()
        permutation = fit_indices[torch.randperm(fit_count, generator=generator)]
        losses = []
        for start in range(0, fit_count, args.batch):
            indices = permutation[start : start + args.batch]
            logits = router.forward_descriptor(descriptors[indices])
            loss = criterion(logits, torch.from_numpy(targets)[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        router.eval()
        with torch.no_grad():
            holdout_scores = router.forward_descriptor(descriptors[holdout_indices]).numpy()
        holdout_metrics = routing_metrics(
            holdout_scores,
            utilities[fit_count:],
            targets[fit_count:],
            args.route_fraction,
        )
        row = {"epoch": epoch, "loss": float(np.mean(losses)), "holdout": holdout_metrics}
        history.append(row)
        key = (holdout_metrics["utility_capture"], -row["loss"])
        if best is None or key > best["key"]:
            best = {"key": key, "epoch": epoch, "state": {k: v.detach().clone() for k, v in router.state_dict().items()}}
            stale = 0
        else:
            stale += 1
        if epoch % 10 == 0 or epoch == 1:
            print(json.dumps(row), flush=True)
        if stale >= args.patience:
            break

    router.load_state_dict(best["state"])
    router.eval()
    with torch.no_grad():
        train_scores = router.forward_descriptor(descriptors).numpy()
        dev_scores = router.forward_descriptor(torch.from_numpy(dev["descriptors"]).float()).numpy()
    train_metrics = routing_metrics(train_scores, utilities, targets, args.route_fraction)
    dev_cutoff = math.ceil(len(dev["utilities"]) * args.route_fraction)
    dev_targets = (dev["ranks"] <= dev_cutoff).astype(np.float32)
    dev_metrics = routing_metrics(dev_scores, dev["utilities"], dev_targets, args.route_fraction)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output.with_suffix(".pt")
    torch.save(
        {
            "state_dict": router.state_dict(),
            "channels": router.channels,
            "bands": router.bands,
            "hidden": 32,
            "route_fraction": args.route_fraction,
        },
        checkpoint_path,
    )
    dev_scores_path = args.output.with_suffix(".dev-scores.jsonl")
    with dev_scores_path.open("w", encoding="utf-8") as output_file:
        for image, score in zip(dev["images"], dev_scores):
            output_file.write(json.dumps({"image": str(image), "score": float(score)}) + "\n")
    report = {
        "status": "frozen router candidate; detection metrics not yet evaluated",
        "protocol": vars(args) | {
            "train_features": str(args.train_features.resolve()),
            "dev_features": str(args.dev_features.resolve()),
            "output": str(args.output.resolve()),
        },
        "parameters": sum(parameter.numel() for parameter in router.parameters()),
        "fit_images": fit_count,
        "holdout_images": len(descriptors) - fit_count,
        "best_epoch": best["epoch"],
        "epochs_run": len(history),
        "train_metrics": train_metrics,
        "dev_metrics": dev_metrics,
        "history": history,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "dev_scores": str(dev_scores_path.resolve()),
        "dev_scores_sha256": hashlib.sha256(dev_scores_path.read_bytes()).hexdigest(),
    }
    args.output.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("parameters", "best_epoch", "epochs_run", "train_metrics", "dev_metrics")}, indent=2))


if __name__ == "__main__":
    main()
