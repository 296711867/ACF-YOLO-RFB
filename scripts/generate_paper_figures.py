"""Generate deterministic paper figures from frozen BDD100K and VisDrone evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_ROOT = Path(os.environ.get("YOLO26_EVIDENCE_ROOT", ROOT)).resolve()
AREAS = ("all", "small", "medium", "large")
AREA_LABELS = ("All", "Small", "Medium", "Large")
CLASS_NAMES = (
    "Pedestrian",
    "People",
    "Bicycle",
    "Car",
    "Van",
    "Truck",
    "Tricycle",
    "Awning tricycle",
    "Bus",
    "Motor",
)
COLORS = ("#4C78A8", "#F58518", "#54A24B", "#B279A2")
def evidence_path(relative_path: str) -> Path:
    """Resolve ignored run evidence from this checkout or a declared evidence root."""
    primary = ROOT / relative_path
    external = EVIDENCE_ROOT / relative_path
    return primary if primary.is_file() or not external.is_file() else external


INPUTS = {
    "bdd_baseline": ROOT / "results/baseline/bdd100k-yolo26n-640-size-metrics.json",
    "bdd_p2": ROOT / "results/p2/bdd100k-yolo26n-p2-640-size-metrics.json",
    "bdd_attention": ROOT / "results/channel-attention-final/bdd100k-p2-channel-attention-e200-size-metrics.json",
    "visdrone_seed0": ROOT / "results/visdrone-native-e100/yolo26n-e100-size-val.json",
    "visdrone_seed1": ROOT / "results/visdrone-multiseed/seed1-size-val.json",
    "visdrone_seed2": ROOT / "results/visdrone-multiseed/seed2-size-val.json",
    "visdrone_summary": ROOT / "results/visdrone-multiseed/summary.json",
    "visdrone_val_official": ROOT / "results/visdrone-native-e100/yolo26n-e100-official-compatible-val.json",
    "visdrone_test_size": ROOT / "results/visdrone-native-e100/yolo26n-e100-size-test-dev.json",
    "visdrone_test_official": ROOT / "results/visdrone-native-e100/yolo26n-e100-official-compatible-test-dev.json",
    "seed0_csv": evidence_path("runs/detect/phase3-visdrone/visdrone-yolo26n-640-e100-stop20-b16-seed0/results.csv"),
    "seed1_csv": evidence_path("runs/detect/phase3-visdrone/visdrone-yolo26n-640-e100-b16-seed1/results.csv"),
    "seed2_csv": evidence_path("runs/detect/phase3-visdrone/visdrone-yolo26n-640-e100-b16-seed2/results.csv"),
}
EXTERNAL_INPUT_KEYS = {"seed0_csv", "seed1_csv", "seed2_csv"}


def sha256(path: Path) -> str:
    """Return one file's SHA256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    """Load one JSON evidence file."""
    return json.loads(path.read_text(encoding="utf-8"))


def configure_style() -> None:
    """Apply a restrained, publication-oriented Matplotlib style."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#D7D7D7",
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, output_dir: Path, name: str) -> list[Path]:
    """Save one figure as deterministic PNG and vector PDF."""
    png = output_dir / f"{name}.png"
    pdf = output_dir / f"{name}.pdf"
    fig.savefig(png, bbox_inches="tight", metadata={"Software": "YOLO26-SDF paper figure generator"})
    fig.savefig(
        pdf,
        bbox_inches="tight",
        metadata={"Creator": "YOLO26-SDF paper figure generator", "CreationDate": None, "ModDate": None},
    )
    plt.close(fig)
    return [png, pdf]


def add_value_labels(ax: plt.Axes, bars, digits: int = 3, errors=None) -> None:
    """Add compact numeric labels above vertical bars."""
    errors = np.zeros(len(bars)) if errors is None else np.asarray(errors)
    for bar, error in zip(bars, errors):
        value = bar.get_height()
        ax.annotate(
            f"{value:.{digits}f}",
            (bar.get_x() + bar.get_width() / 2, value + error),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
        )


def plot_bdd_size_comparison(output_dir: Path) -> list[Path]:
    """Compare BDD100K size AP across retained and rejected full models."""
    sources = ("bdd_baseline", "bdd_p2", "bdd_attention")
    labels = ("YOLO26n", "YOLO26n-P2", "P2 + channel attention")
    metrics = [load_json(INPUTS[source])["metrics"] for source in sources]
    x = np.arange(len(AREAS))
    width = 0.24
    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    for index, (label, metric, color) in enumerate(zip(labels, metrics, COLORS)):
        values = [metric[area]["AP50_95"] for area in AREAS]
        bars = ax.bar(x + (index - 1) * width, values, width, label=label, color=color)
        add_value_labels(ax, bars)
    ax.set_ylabel("AP50-95")
    ax.set_xticks(x, AREA_LABELS)
    ax.set_ylim(0, 0.58)
    ax.set_title("BDD100K: object-size AP at 640 px")
    ax.legend(loc="upper left", ncols=3)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return save_figure(fig, output_dir, "bdd-size-ap-comparison")


def load_training_curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load epoch and mAP50-95 arrays from an Ultralytics results CSV."""
    with path.open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    return (
        np.asarray([int(row["epoch"]) for row in rows]),
        np.asarray([float(row["metrics/mAP50-95(B)"]) for row in rows]),
    )


def plot_visdrone_training_curves(output_dir: Path) -> list[Path]:
    """Plot three frozen VisDrone training trajectories."""
    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    annotations = {0: ((4, 5), "left"), 1: ((4, 13), "left"), 2: ((-4, 21), "right")}
    for seed, color in zip((0, 1, 2), COLORS):
        epochs, values = load_training_curve(INPUTS[f"seed{seed}_csv"])
        best_index = int(np.argmax(values))
        ax.plot(epochs, values, color=color, linewidth=1.5, label=f"Seed {seed}")
        ax.scatter(epochs[best_index], values[best_index], color=color, s=22, zorder=3)
        ax.annotate(
            f"e{epochs[best_index]}: {values[best_index]:.3f}",
            (epochs[best_index], values[best_index]),
            xytext=annotations[seed][0],
            textcoords="offset points",
            ha=annotations[seed][1],
            fontsize=7,
            color="#333333",
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation mAP50-95")
    ax.set_xlim(1, 100)
    ax.set_ylim(0.04, 0.185)
    ax.set_title("VisDrone: frozen YOLO26n training across seeds")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return save_figure(fig, output_dir, "visdrone-training-curves")


def plot_visdrone_multiseed_errorbars(output_dir: Path) -> list[Path]:
    """Plot multi-seed size AP and AR means with sample-standard-deviation bars."""
    summary = load_json(INPUTS["visdrone_summary"])["aggregate"]["size"]
    x = np.arange(len(AREAS))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8), sharex=True)
    for ax, key, title, color in (
        (axes[0], "AP50_95", "Average precision", COLORS[0]),
        (axes[1], "AR50_95", "Average recall", COLORS[1]),
    ):
        means = [summary[area][key]["mean"] for area in AREAS]
        errors = [summary[area][key]["sample_std"] for area in AREAS]
        bars = ax.bar(x, means, yerr=errors, capsize=4, color=color, alpha=0.88)
        add_value_labels(ax, bars, errors=errors)
        ax.set_xticks(x, AREA_LABELS)
        ax.set_ylabel(key.replace("_", "-"))
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, max(np.asarray(means) + np.asarray(errors)) * 1.22)
    fig.suptitle("VisDrone: three-seed mean ± sample standard deviation", y=1.01, fontsize=10)
    fig.tight_layout()
    return save_figure(fig, output_dir, "visdrone-multiseed-size-errorbars")


def plot_visdrone_val_test_gap(output_dir: Path) -> list[Path]:
    """Show the frozen seed0 internal size AP generalization gap."""
    val = load_json(INPUTS["visdrone_seed0"])["metrics"]
    test = load_json(INPUTS["visdrone_test_size"])["metrics"]
    x = np.arange(len(AREAS))
    width = 0.34
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    val_bars = ax.bar(x - width / 2, [val[a]["AP50_95"] for a in AREAS], width, label="Val", color=COLORS[0])
    test_bars = ax.bar(
        x + width / 2, [test[a]["AP50_95"] for a in AREAS], width, label="Test-dev (frozen once)", color=COLORS[1]
    )
    add_value_labels(ax, val_bars)
    add_value_labels(ax, test_bars)
    declines = [(test[a]["AP50_95"] / val[a]["AP50_95"] - 1) * 100 for a in AREAS]
    for index, decline in enumerate(declines):
        y = max(val[AREAS[index]]["AP50_95"], test[AREAS[index]]["AP50_95"]) + 0.035
        ax.text(index, y, f"{decline:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("AP50-95")
    ax.set_xticks(x, AREA_LABELS)
    ax.set_ylim(0, 0.5)
    ax.set_title("VisDrone seed 0: val-to-test object-size gap")
    ax.legend(loc="upper left")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return save_figure(fig, output_dir, "visdrone-val-test-size-gap")


def plot_visdrone_class_gap(output_dir: Path) -> list[Path]:
    """Compare official-compatible class AP between val and frozen test-dev."""
    val = load_json(INPUTS["visdrone_val_official"])["metrics"]["per_class"]
    test = load_json(INPUTS["visdrone_test_official"])["metrics"]["per_class"]
    y = np.arange(len(CLASS_NAMES))
    height = 0.34
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    val_values = [val[str(index)]["AP50_95"] for index in range(1, 11)]
    test_values = [test[str(index)]["AP50_95"] for index in range(1, 11)]
    ax.barh(y - height / 2, val_values, height, label="Val", color=COLORS[0])
    ax.barh(y + height / 2, test_values, height, label="Test-dev (frozen once)", color=COLORS[1])
    ax.set_yticks(y, CLASS_NAMES)
    ax.invert_yaxis()
    ax.set_xlabel("Official-compatible AP50-95 (%)")
    ax.set_title("VisDrone seed 0: per-class generalization")
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return save_figure(fig, output_dir, "visdrone-official-class-gap")


def main() -> None:
    """Generate all frozen paper figures and a hash manifest."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/paper_figures"))
    args = parser.parse_args()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    outputs = []
    for plotter in (
        plot_bdd_size_comparison,
        plot_visdrone_training_curves,
        plot_visdrone_multiseed_errorbars,
        plot_visdrone_val_test_gap,
        plot_visdrone_class_gap,
    ):
        outputs.extend(plotter(output_dir))
    manifest = {
        "generator": str(Path(__file__).resolve().relative_to(ROOT)),
        "inputs": {
            name: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)} for name, path in INPUTS.items()
        },
        "outputs": {path.name: sha256(path) for path in outputs},
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"figures": len(outputs) // 2, "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
