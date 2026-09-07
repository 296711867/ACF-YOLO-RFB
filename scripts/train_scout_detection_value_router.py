"""Train one frozen ScoutDet-RV seed and compare strong scout-only baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from torch.utils.data import DataLoader, TensorDataset

from scripts.evaluate_resolution_router import hash_random_scores
from scripts.train_perspective_router import routing_metrics
from sdf.refinement import ScoutDetectionValueRouter, scout_value_loss


BASELINE_FEATURES = (
    "count_fraction", "confidence_sum", "confidence_mean", "confidence_max",
    "small_count_fraction", "small_confidence_sum", "low_confidence_fraction",
    "class_entropy", "crowded_cell_fraction", "area_mean", "area_std", "center_y_mean",
    "band0_fraction", "band1_fraction", "band2_fraction", "band3_fraction",
)


def load_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def sequence_order(sequences: np.ndarray) -> list[str]:
    return sorted({str(value) for value in sequences}, key=lambda value: hashlib.sha256(f"p14:{value}".encode()).digest())


def aggregate_scout_statistics(tokens: np.ndarray, classes: np.ndarray, mask: np.ndarray, num_classes: int = 10) -> np.ndarray:
    output = np.zeros((len(tokens), len(BASELINE_FEATURES)), dtype=np.float32)
    for index, (sample, class_ids, valid) in enumerate(zip(tokens, classes, mask)):
        values = sample[valid]
        ids = class_ids[valid].astype(int)
        if not len(values):
            continue
        confidence, area, center_y = values[:, 4], values[:, 5], values[:, 1]
        small = area < 0.04
        histogram = np.bincount(ids, minlength=num_classes).astype(np.float32)
        probabilities = histogram[histogram > 0] / len(ids)
        entropy = float(-(probabilities * np.log(probabilities)).sum() / math.log(num_classes))
        grid_x = np.clip((values[:, 0] * 4).astype(int), 0, 3)
        grid_y = np.clip((center_y * 4).astype(int), 0, 3)
        occupancy = np.bincount(grid_y * 4 + grid_x, minlength=16)
        bands = np.bincount(grid_y, minlength=4) / len(values)
        output[index] = (
            len(values) / sample.shape[0], confidence.sum(), confidence.mean(), confidence.max(),
            small.mean(), confidence[small].sum(), (confidence < 0.1).mean(), entropy,
            np.sum(occupancy > 1) / 16.0, area.mean(), area.std(), center_y.mean(), *bands,
        )
    return output


def ridge_predict(fit: np.ndarray, targets: np.ndarray, query: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    mean, std = fit.mean(axis=0), fit.std(axis=0)
    std[std < 1e-8] = 1.0
    fit_scaled, query_scaled = (fit - mean) / std, (query - mean) / std
    design = np.column_stack((fit_scaled, np.ones(len(fit_scaled))))
    query_design = np.column_stack((query_scaled, np.ones(len(query_scaled))))
    penalty = np.eye(design.shape[1], dtype=np.float64) * alpha
    penalty[-1, -1] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ targets)
    return query_design @ coefficients


def fixed_baseline_scores(
    fit_statistics: np.ndarray,
    fit_utilities: np.ndarray,
    holdout_statistics: np.ndarray,
    names: list[str],
) -> dict[str, np.ndarray]:
    hgb = HistGradientBoostingRegressor(
        learning_rate=0.05, max_iter=100, max_leaf_nodes=15, min_samples_leaf=20,
        l2_regularization=1.0, random_state=0,
    ).fit(fit_statistics, fit_utilities)
    count = holdout_statistics[:, 0] * 192.0
    confidence_mean = holdout_statistics[:, 2]
    return {
        "hash_random": hash_random_scores(names, seed=0),
        "detection_count": count,
        "small_confidence_sum": holdout_statistics[:, 5],
        "uncertainty_crowding": holdout_statistics[:, 5] + 0.05 * (1.0 - confidence_mean) * count + 0.5 * holdout_statistics[:, 8],
        "ridge_statistics": ridge_predict(fit_statistics, fit_utilities, holdout_statistics),
        "hist_gradient_boosting": hgb.predict(holdout_statistics),
    }


def score_model(model: ScoutDetectionValueRouter, cache: dict[str, np.ndarray], indices: np.ndarray, batch: int, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    means, log_variances = [], []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), batch):
            selected = indices[start:start + batch]
            output = model(
                torch.from_numpy(cache["tokens"][selected]).to(device),
                torch.from_numpy(cache["classes"][selected].astype(np.int64)).to(device),
                torch.from_numpy(cache["mask"][selected]).to(device),
            )
            means.append(output[0].cpu())
            log_variances.append(output[1].cpu())
    return torch.cat(means).numpy(), torch.cat(log_variances).numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--route-fraction", type=float, default=0.25)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cache = load_cache(args.cache)
    groups = sequence_order(cache["sequences"])
    if len(groups) != 208:
        raise ValueError(f"frozen protocol requires 208 sequence groups, received {len(groups)}")
    optimize_groups, select_groups, holdout_groups = set(groups[:128]), set(groups[128:160]), set(groups[160:])
    sequences = np.asarray([str(value) for value in cache["sequences"]])
    optimize = np.flatnonzero(np.isin(sequences, list(optimize_groups)))
    selection = np.flatnonzero(np.isin(sequences, list(select_groups)))
    holdout = np.flatnonzero(np.isin(sequences, list(holdout_groups)))
    if set(cache["names"][optimize]) & set(cache["names"][holdout]):
        raise ValueError("fit and holdout images overlap")
    utilities = cache["utilities"].astype(np.float32)
    utility_mean, utility_std = float(utilities[optimize].mean()), float(utilities[optimize].std())
    if utility_std <= 0 or not math.isfinite(utility_std):
        raise ValueError("optimization utilities need positive finite variance")
    device = torch.device("cuda:0" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    dataset = TensorDataset(
        torch.from_numpy(cache["tokens"][optimize]),
        torch.from_numpy(cache["classes"][optimize].astype(np.int64)),
        torch.from_numpy(cache["mask"][optimize]),
        torch.from_numpy(utilities[optimize]),
    )
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, generator=torch.Generator().manual_seed(args.seed))
    model = ScoutDetectionValueRouter().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best, stale, history = None, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for tokens, classes, mask, batch_utilities in loader:
            means, log_variances = model(tokens.to(device), classes.to(device), mask.to(device))
            loss, _ = scout_value_loss(means, log_variances, batch_utilities.to(device), utility_mean, utility_std)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        selection_means, _ = score_model(model, cache, selection, args.batch, device)
        selection_targets = (utilities[selection] > 0).astype(np.float32)
        metrics = routing_metrics(selection_means, utilities[selection], selection_targets, args.route_fraction)
        row = {"epoch": epoch, "loss": float(np.mean(losses)), "selection": metrics}
        history.append(row)
        key = (metrics["utility_capture"], -row["loss"])
        if best is None or key > best["key"]:
            best = {"key": key, "epoch": epoch, "state": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}}
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(json.dumps(row), flush=True)
        if stale >= args.patience:
            break
    model.load_state_dict(best["state"])
    model.to(device)
    holdout_means, holdout_log_variances = score_model(model, cache, holdout, args.batch, device)
    holdout_utilities = utilities[holdout]
    targets = (holdout_utilities > 0).astype(np.float32)
    neural_metrics = routing_metrics(holdout_means, holdout_utilities, targets, args.route_fraction)
    statistics = aggregate_scout_statistics(cache["tokens"], cache["classes"], cache["mask"])
    fit = np.concatenate((optimize, selection))
    baseline_scores = fixed_baseline_scores(statistics[fit], utilities[fit], statistics[holdout], [str(name) for name in cache["names"][holdout]])
    baselines = {name: routing_metrics(scores, holdout_utilities, targets, args.route_fraction) for name, scores in baseline_scores.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output.with_suffix(".pt")
    torch.save({"state_dict": best["state"], "seed": args.seed, "protocol": "P14-SCOUT-DETECTION-VALUE-ROUTER"}, checkpoint)
    score_path = args.output.with_suffix(".holdout-scores.jsonl")
    with score_path.open("w", encoding="utf-8") as output_file:
        for name, mean, log_variance, utility in zip(cache["names"][holdout], holdout_means, holdout_log_variances, holdout_utilities):
            output_file.write(json.dumps({"image": str(name), "mean": float(mean), "log_variance": float(log_variance), "utility": float(utility)}) + "\n")
    report = {
        "status": "P14 train internal holdout only; val and test-dev excluded",
        "seed": args.seed,
        "device": str(device),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "groups": {"optimize": 128, "selection": 32, "holdout": 48},
        "images": {"optimize": len(optimize), "selection": len(selection), "holdout": len(holdout)},
        "best_epoch": best["epoch"], "epochs_run": len(history),
        "neural": neural_metrics, "baselines": baselines,
        "baseline_parameters": {"features": list(BASELINE_FEATURES), "ridge_alpha": 1.0, "hist_gradient_boosting": {"learning_rate": 0.05, "max_iter": 100, "max_leaf_nodes": 15, "min_samples_leaf": 20, "l2_regularization": 1.0, "random_state": 0}},
        "checkpoint": str(checkpoint.resolve()), "scores": str(score_path.resolve()), "history": history,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("seed", "best_epoch", "epochs_run", "neural", "baselines")}, indent=2))


if __name__ == "__main__":
    main()
