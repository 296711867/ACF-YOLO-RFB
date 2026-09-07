"""Train the frozen detector-free thumbnail utility router."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from scripts.evaluate_resolution_router import hash_random_scores
from scripts.train_perspective_router import routing_metrics, top_fraction_indices
from sdf.refinement import ThumbnailUtilityRouter, utility_routing_loss


class ThumbnailDataset(Dataset):
    """Keep compressed-cache images as uint8 and convert only requested batches."""

    def __init__(self, images: np.ndarray, utilities: np.ndarray, indices: np.ndarray) -> None:
        self.images = images
        self.utilities = utilities
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        source_index = int(self.indices[index])
        image = np.ascontiguousarray(self.images[source_index].transpose(2, 0, 1))
        return torch.from_numpy(image).float().div_(255.0), torch.tensor(self.utilities[source_index])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def ridge_scores(
    fit_statistics: np.ndarray,
    fit_utilities: np.ndarray,
    target_statistics: np.ndarray,
    regularization: float = 1e-3,
) -> np.ndarray:
    """Fit one frozen standardized ridge utility baseline with an unpenalized bias."""
    mean = fit_statistics.mean(axis=0)
    std = fit_statistics.std(axis=0)
    std[std < 1e-8] = 1.0
    fit = (fit_statistics - mean) / std
    target = (target_statistics - mean) / std
    design = np.concatenate((fit, np.ones((len(fit), 1), dtype=np.float32)), axis=1)
    target_design = np.concatenate((target, np.ones((len(target), 1), dtype=np.float32)), axis=1)
    penalty = np.eye(design.shape[1], dtype=np.float32) * regularization
    penalty[-1, -1] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ fit_utilities)
    return (target_design @ coefficients).astype(np.float32)


def baseline_scores(train: dict[str, np.ndarray], dev: dict[str, np.ndarray], fit_count: int, seed: int) -> dict[str, np.ndarray]:
    """Return all frozen same-cost thumbnail baseline scores on the dev cache."""
    statistic_names = [str(value) for value in train["statistic_names"]]
    if statistic_names != [str(value) for value in dev["statistic_names"]]:
        raise ValueError("train and dev statistic schemas differ")
    columns = {name: index for index, name in enumerate(statistic_names)}
    statistics = dev["statistics"]
    return {
        "hash_random": hash_random_scores([str(value) for value in dev["names"]], seed),
        "gray_std": statistics[:, columns["gray_std"]],
        "laplacian_abs_mean": statistics[:, columns["laplacian_abs_mean"]],
        "sobel_magnitude_mean": statistics[:, columns["sobel_magnitude_mean"]],
        "upper_band_laplacian": (
            statistics[:, columns["band0_laplacian_abs_mean"]]
            + statistics[:, columns["band1_laplacian_abs_mean"]]
        )
        / 2.0,
        "ridge_statistics": ridge_scores(
            train["statistics"][:fit_count],
            train["utilities"][:fit_count],
            statistics,
        ),
    }


def top_utility_targets(utilities: np.ndarray, fraction: float) -> np.ndarray:
    """Build local exact-budget targets without relying on ranks from a larger set."""
    targets = np.zeros(len(utilities), dtype=np.float32)
    targets[top_fraction_indices(utilities, fraction)] = 1.0
    return targets


def score_model(
    model: ThumbnailUtilityRouter,
    images: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Score cached thumbnails without copying the full cache to float memory."""
    model.eval()
    output = []
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            batch = np.ascontiguousarray(images[start : start + batch_size].transpose(0, 3, 1, 2))
            inputs = torch.from_numpy(batch).to(device=device, dtype=torch.float32).div_(255.0)
            output.append(model(inputs).cpu())
    return torch.cat(output).numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--dev-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--route-fraction", type=float, default=0.25)
    parser.add_argument("--fit-count", type=int, default=4000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--regression-weight", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    if not 0 < args.route_fraction <= 1:
        raise ValueError("route_fraction must be in (0, 1]")
    if args.epochs <= 0 or args.patience <= 0 or args.batch <= 1:
        raise ValueError("epochs, patience, and batch must be positive; batch must exceed one")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train = load_cache(args.train_cache)
    dev = load_cache(args.dev_cache)
    if not 1 < args.fit_count < len(train["images"]):
        raise ValueError("fit_count must leave a non-empty internal holdout")
    if train["images"].shape[1:] != (96, 160, 3) or dev["images"].shape[1:] != (96, 160, 3):
        raise ValueError("thumbnail caches must use frozen 160x96 RGB inputs")
    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    fit_indices = np.arange(args.fit_count)
    holdout_indices = np.arange(args.fit_count, len(train["images"]))
    utility_mean = float(train["utilities"][: args.fit_count].mean())
    utility_std = float(train["utilities"][: args.fit_count].std())
    if not math.isfinite(utility_std) or utility_std <= 0:
        raise ValueError("fit utilities must have positive finite standard deviation")

    dataset = ThumbnailDataset(train["images"], train["utilities"], fit_indices)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        generator=generator,
        drop_last=False,
    )
    model = ThumbnailUtilityRouter().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = None
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        for inputs, utilities in loader:
            inputs = inputs.to(device)
            utilities = utilities.to(device)
            scores = model(inputs)
            loss, components = utility_routing_loss(
                scores,
                utilities,
                utility_mean,
                utility_std,
                args.regression_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_losses.append(
                {
                    "total": float(loss.detach()),
                    "ranking": float(components["ranking"].detach()),
                    "regression": float(components["regression"].detach()),
                }
            )

        holdout_scores = score_model(model, train["images"][holdout_indices], args.batch, device)
        holdout_utilities = train["utilities"][holdout_indices]
        holdout_targets = top_utility_targets(holdout_utilities, args.route_fraction)
        holdout_metrics = routing_metrics(
            holdout_scores,
            holdout_utilities,
            holdout_targets,
            args.route_fraction,
        )
        row = {
            "epoch": epoch,
            "loss": {key: float(np.mean([item[key] for item in epoch_losses])) for key in epoch_losses[0]},
            "holdout": holdout_metrics,
        }
        history.append(row)
        key = (holdout_metrics["utility_capture"], -row["loss"]["total"])
        if best is None or key > best["key"]:
            best = {
                "key": key,
                "epoch": epoch,
                "state": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
            }
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(json.dumps(row), flush=True)
        if stale >= args.patience:
            break

    model.load_state_dict(best["state"])
    model.to(device)
    train_scores = score_model(model, train["images"], args.batch, device)
    dev_scores = score_model(model, dev["images"], args.batch, device)
    train_targets = top_utility_targets(train["utilities"], args.route_fraction)
    dev_targets = top_utility_targets(dev["utilities"], args.route_fraction)
    train_metrics = routing_metrics(train_scores, train["utilities"], train_targets, args.route_fraction)
    dev_metrics = routing_metrics(dev_scores, dev["utilities"], dev_targets, args.route_fraction)

    baseline_values = baseline_scores(train, dev, args.fit_count, args.seed)
    baselines = {
        name: routing_metrics(scores, dev["utilities"], dev_targets, args.route_fraction)
        for name, scores in baseline_values.items()
    }
    image_baseline_names = [name for name in baselines if name != "hash_random"]
    best_baseline_name = max(image_baseline_names, key=lambda name: baselines[name]["utility_capture"])
    best_baseline_capture = baselines[best_baseline_name]["utility_capture"]
    relative_gain = (
        (dev_metrics["utility_capture"] / best_baseline_capture - 1) * 100 if best_baseline_capture else None
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output.with_suffix(".pt")
    torch.save(
        {
            "state_dict": best["state"],
            "channels": model.channels,
            "bands": model.bands,
            "hidden": model.hidden,
            "thumbnail_width": 160,
            "thumbnail_height": 96,
            "route_fraction": args.route_fraction,
            "utility_mean": utility_mean,
            "utility_std": utility_std,
        },
        checkpoint_path,
    )
    scores_path = args.output.with_suffix(".dev-scores.jsonl")
    with scores_path.open("w", encoding="utf-8") as output_file:
        for name, score in zip(dev["names"], dev_scores):
            output_file.write(json.dumps({"image": str(name), "score": float(score)}) + "\n")

    protocol = vars(args).copy()
    for key in ("train_cache", "dev_cache", "output"):
        protocol[key] = str(Path(protocol[key]).resolve())
    report = {
        "status": "P12 development result; not formal BDD val or external evidence",
        "protocol": protocol,
        "device": str(device),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "fit_images": args.fit_count,
        "holdout_images": len(holdout_indices),
        "dev_images": len(dev["images"]),
        "utility_mean": utility_mean,
        "utility_std": utility_std,
        "best_epoch": best["epoch"],
        "epochs_run": len(history),
        "train_metrics": train_metrics,
        "dev_metrics": dev_metrics,
        "dev_baselines": baselines,
        "best_dev_image_baseline": best_baseline_name,
        "relative_utility_capture_percent": relative_gain,
        "history": history,
        "train_cache_sha256": sha256(args.train_cache),
        "dev_cache_sha256": sha256(args.dev_cache),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256(checkpoint_path),
        "dev_scores": str(scores_path.resolve()),
        "dev_scores_sha256": sha256(scores_path),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "parameters": report["parameters"],
                "best_epoch": report["best_epoch"],
                "dev_metrics": dev_metrics,
                "best_dev_image_baseline": best_baseline_name,
                "best_dev_image_baseline_metrics": baselines[best_baseline_name],
                "relative_utility_capture_percent": relative_gain,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
