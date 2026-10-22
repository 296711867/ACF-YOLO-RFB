"""Multi-seed paired image bootstrap: pool RFB-vs-L0 deltas across seeds.

P29 protocol addendum (RFB-3S reopen). Each replicate draws ONE image
multinomial weight vector (the val set is the same 10k images for every seed)
and applies it to every (L0_s, RFB_s) pair; the pooled statistic is the mean of
per-seed deltas. Image-sampling uncertainty is resampled; seed noise is
averaged. Reuses the exact AP machinery of bootstrap_bdd_paired_ap.py.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

from scripts.bootstrap_bdd_paired_ap import (
    AREAS,
    PreparedModel,
    load_prediction_records,
    model_ap,
    prepare_model,
    sha256,
)

_MULTISEED_STATE = None


def _bootstrap_one(seed: int) -> dict[str, dict[str, float]]:
    """Evaluate one image-resample across all seed pairs using inherited state."""
    if _MULTISEED_STATE is None:
        raise RuntimeError("Multi-seed worker state has not been initialized")
    pairs, image_count = _MULTISEED_STATE
    rng = np.random.default_rng(seed)
    weights = np.bincount(rng.integers(0, image_count, size=image_count), minlength=image_count).astype(np.float64)
    output: dict[str, dict[str, float]] = {}
    for area_name in AREAS:
        per_seed_delta = []
        for baseline, candidate in pairs:
            baseline_ap = model_ap(baseline, weights, area_name)
            candidate_ap = model_ap(candidate, weights, area_name)
            per_seed_delta.append(candidate_ap - baseline_ap)
        output[area_name] = {"pooled_delta": float(np.mean(per_seed_delta))}
    return output


def _observed(pairs: list[tuple[PreparedModel, PreparedModel]], image_count: int) -> dict:
    """Unweighted per-seed APs and deltas plus the pooled mean."""
    ones = np.ones(image_count, dtype=np.float64)
    per_seed = {area: [] for area in AREAS}
    for index, (baseline, candidate) in enumerate(pairs):
        for area_name in AREAS:
            baseline_ap = model_ap(baseline, ones, area_name)
            candidate_ap = model_ap(candidate, ones, area_name)
            per_seed[area_name].append(
                {
                    "seed_pair_index": index,
                    "baseline_AP50_95": baseline_ap,
                    "candidate_AP50_95": candidate_ap,
                    "delta_AP50_95": candidate_ap - baseline_ap,
                }
            )
    return {
        area_name: {
            "per_seed": entries,
            "pooled_delta_mean": float(np.mean([entry["delta_AP50_95"] for entry in entries])),
        }
        for area_name, entries in per_seed.items()
    }


def multiseed_bootstrap(
    pairs: list[tuple[PreparedModel, PreparedModel]],
    image_count: int,
    replicates: int,
    seed: int,
    workers: int,
) -> dict:
    """Run the pooled paired bootstrap for all and small AP."""
    global _MULTISEED_STATE
    _MULTISEED_STATE = (pairs, image_count)
    seeds = np.random.SeedSequence(seed).generate_state(replicates).tolist()
    if workers == 1:
        results = map(_bootstrap_one, seeds)
    else:
        context = mp.get_context("fork")
        with context.Pool(processes=workers) as pool:
            results = pool.map(_bootstrap_one, seeds)
    pooled = {area: [] for area in AREAS}
    for result in results:
        for area_name, values in result.items():
            pooled[area_name].append(values["pooled_delta"])
    output = {}
    for area_name, values in pooled.items():
        array = np.asarray(values)
        output[area_name] = {
            "delta": {
                "mean": float(np.mean(array)),
                "ci95_percentile": [float(value) for value in np.percentile(array, [2.5, 97.5])],
            },
            "probability_delta_gt_zero": float(np.mean(array > 0)),
        }
    return output


def parse_args() -> argparse.Namespace:
    """Parse baseline=candidate prediction pair paths."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pair",
        action="append",
        required=True,
        metavar="BASELINE=CANDIDATE",
        help="repeatable baseline=candidate paths to frozen predictions.jsonl",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=260805)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    """Load pairs, verify a shared image set, run the pooled bootstrap, write JSON."""
    args = parse_args()
    pairs: list[tuple[PreparedModel, PreparedModel]] = []
    reference_keys = None
    provenance = []
    for spec in args.pair:
        baseline_path, candidate_path = (Path(side) for side in spec.split("=", 1))
        baseline_keys, baseline_records = load_prediction_records(baseline_path)
        candidate_keys, candidate_records = load_prediction_records(candidate_path)
        if baseline_keys != candidate_keys:
            raise SystemExit(f"image key mismatch within pair: {spec}")
        if reference_keys is None:
            reference_keys = baseline_keys
        elif baseline_keys != reference_keys:
            raise SystemExit("image key mismatch across pairs (all seeds must share the val set)")
        pairs.append((prepare_model(baseline_records), prepare_model(candidate_records)))
        provenance.append(
            {
                "baseline": str(baseline_path),
                "candidate": str(candidate_path),
                "baseline_sha256": sha256(baseline_path),
                "candidate_sha256": sha256(candidate_path),
            }
        )
    image_count = len(reference_keys)
    bootstrap = multiseed_bootstrap(pairs, image_count, args.replicates, args.seed, args.workers)
    payload = {
        "method": "pooled paired image bootstrap across seeds (shared image resample per replicate)",
        "replicates": args.replicates,
        "seed": args.seed,
        "n_seed_pairs": len(pairs),
        "image_count": image_count,
        "provenance": provenance,
        "observed": _observed(pairs, image_count),
        "bootstrap": bootstrap,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "pooled": {a: bootstrap[a] for a in AREAS}}, indent=2))


if __name__ == "__main__":
    main()
