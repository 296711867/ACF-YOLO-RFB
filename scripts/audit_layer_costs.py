"""Per-layer parameter/MACs audit for the Chapter-5 lightweight study (P20).

Profiles one DetectionModel YAML at 640x640 with forward hooks on every
Conv2d/Linear leaf, attributes compute to the top-level layer that owns it,
classifies each top-level layer into the frozen path taxonomy, and projects
the cost of replacing high-level plain Conv layers with DWConv or GhostConv.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn
from ultralytics.nn.modules.conv import Conv, DWConv, GhostConv
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import get_flops, get_num_params

ROOT = Path(__file__).resolve().parents[1]

# Frozen path taxonomy for the yolo26n-p2-acr-acfr layout. Protected paths are
# the small-object-sensitive ones; only plain Conv layers inside the two
# high-level classes are eligible for the unit swap.
STRIDE_TO_PATH = {4: "P2 (protected)", 8: "P3 (protected)", 16: "P4 high-level", 32: "P5 high-level"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-yaml", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--label", type=str, default="L0")
    return parser.parse_args()


def top_level_index(name: str) -> int:
    """Map a submodule name like 'model.7.cv1.conv' to top-level layer 7."""
    parts = name.split(".")
    return int(parts[1])


def conv_macs(module: nn.Conv2d, output: torch.Tensor) -> int:
    """Multiply-accumulate count for one Conv2d forward."""
    out_h, out_w = output.shape[-2:]
    return (
        (module.in_channels // module.groups)
        * module.out_channels
        * module.kernel_size[0]
        * module.kernel_size[1]
        * out_h
        * out_w
    )


def linear_macs(module: nn.Linear, output: torch.Tensor) -> int:
    """Multiply-accumulate count for one Linear forward over all spatial tokens."""
    return module.in_features * module.out_features * output.numel() // output.shape[-1]


def dwconv_cost(c1: int, c2: int, stride: int, out_h: int, out_w: int) -> tuple[int, int]:
    """Params and MACs of DWConv(c1, c2, 3, stride); groups = gcd(c1, c2)."""
    groups = math.gcd(c1, c2)
    macs_per_position = (c1 // groups) * c2 * 9
    macs = macs_per_position * out_h * out_w
    params = macs_per_position + 2 * c2  # conv weight + BN affine
    return params, macs


def ghostconv_cost(c1: int, c2: int, stride: int, out_h: int, out_w: int) -> tuple[int, int]:
    """Params and MACs of GhostConv(c1, c2, 3, stride): 3x3 primary + 5x5 depthwise."""
    hidden = c2 // 2
    cv1_macs = c1 * hidden * 9 * out_h * out_w
    cv2_macs = 25 * hidden * out_h * out_w
    macs = cv1_macs + cv2_macs
    params = c1 * hidden * 9 + 2 * hidden + 25 * hidden + 2 * hidden
    return params, macs


def profile(yaml_path: Path, imgsz: int) -> dict:
    torch.manual_seed(0)
    model = DetectionModel(str(yaml_path), ch=3, nc=80, verbose=False)
    model.eval()
    image = torch.zeros(1, 3, imgsz, imgsz)

    layer_macs: dict[int, int] = defaultdict(int)
    layer_params: dict[int, int] = {}
    layer_specs: dict[int, list] = defaultdict(list)

    def conv_hook(name: str):
        def hook(module, inputs, output):
            layer_macs[top_level_index(name)] += conv_macs(module, output)
            layer_specs[top_level_index(name)].append(
                {
                    "kind": "conv2d",
                    "in_channels": module.in_channels,
                    "out_channels": module.out_channels,
                    "kernel": list(module.kernel_size),
                    "stride": list(module.stride),
                    "groups": module.groups,
                }
            )

        return hook

    def linear_hook(name: str):
        def hook(module, inputs, output):
            layer_macs[top_level_index(name)] += linear_macs(module, output)
            layer_specs[top_level_index(name)].append(
                {"kind": "linear", "in_features": module.in_features, "out_features": module.out_features}
            )

        return hook

    handles = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(conv_hook(name)))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_hook(name)))
    with torch.no_grad():
        model(image)
    for handle in handles:
        handle.remove()

    for index, layer in enumerate(model.model):
        layer_params[index] = sum(p.numel() for p in layer.parameters())

    total_params = sum(p.numel() for p in model.parameters())
    total_macs = sum(layer_macs.values())
    gflops = get_flops(model, imgsz)

    rows = []
    for index, layer in enumerate(model.model):
        rows.append(
            {
                "layer": index,
                "module": type(layer).__name__,
                "params": layer_params[index],
                "macs": layer_macs[index],
                "macs_share": layer_macs[index] / total_macs,
                "top_level_ops": layer_specs[index],
            }
        )
    return {
        "model_yaml": str(yaml_path),
        "imgsz": imgsz,
        "layers": rows,
        "total_params": total_params,
        "total_macs": total_macs,
        "total_gmacs": total_macs / 1e9,
        "ultralytics_info": {"n_params": get_num_params(model), "gflops": gflops},
    }


def main() -> None:
    args = parse_args()
    audit = profile(args.model_yaml, args.imgsz)

    # Recover per-layer output spatial size by replaying with shape hooks.
    torch.manual_seed(0)
    model = DetectionModel(str(args.model_yaml), ch=3, nc=80, verbose=False)
    model.eval()
    out_shapes: dict[int, tuple] = {}

    def shape_hook(index: int):
        def hook(module, inputs, output):
            if isinstance(output, torch.Tensor):
                out_shapes[index] = tuple(output.shape)
            elif isinstance(output, (list, tuple)) and output and isinstance(output[0], torch.Tensor):
                out_shapes[index] = tuple(output[0].shape)

        return hook

    handles = [layer.register_forward_hook(shape_hook(i)) for i, layer in enumerate(model.model)]
    with torch.no_grad():
        model(torch.zeros(1, 3, args.imgsz, args.imgsz))
    for handle in handles:
        handle.remove()

    for row in audit["layers"]:
        index = row["layer"]
        shape = out_shapes.get(index)
        if shape is not None and len(shape) == 4:
            row["out_shape"] = list(shape)
            row["out_stride"] = args.imgsz // shape[-1]
            row["out_channels"] = shape[1]
            row["path"] = STRIDE_TO_PATH.get(row["out_stride"], f"stride {row['out_stride']}")
        else:
            row["out_shape"] = None
            row["out_stride"] = None
            row["out_channels"] = None
            row["path"] = "non-spatial / head"
        if row["module"] == "Detect":
            row["path"] = "Detect head (immutable)"

    # Candidate set: plain top-level Conv layers whose output stride >= 16.
    candidate_layers = []
    for row in audit["layers"]:
        if row["module"] == "Conv" and row["out_stride"] in (16, 32):
            candidate_layers.append(row)

    projection = {"candidate_positions": []}
    drop_params = {"DWConv": 0, "GhostConv": 0}
    drop_macs = {"DWConv": 0, "GhostConv": 0}
    for row in candidate_layers:
        spec = row["top_level_ops"][0]
        c1, c2 = spec["in_channels"], spec["out_channels"]
        out_h, out_w = row["out_shape"][-2], row["out_shape"][-1]
        dw_params, dw_macs = dwconv_cost(c1, c2, spec["stride"][0], out_h, out_w)
        gh_params, gh_macs = ghostconv_cost(c1, c2, spec["stride"][0], out_h, out_w)
        projection["candidate_positions"].append(
            {
                "layer": row["layer"],
                "path": row["path"],
                "in_channels": c1,
                "out_channels": c2,
                "out_shape": row["out_shape"],
                "original": {"params": row["params"], "macs": row["macs"]},
                "dwconv": {"params": dw_params, "macs": dw_macs},
                "ghostconv": {"params": gh_params, "macs": gh_macs},
            }
        )
        drop_params["DWConv"] += row["params"] - dw_params
        drop_params["GhostConv"] += row["params"] - gh_params
        drop_macs["DWConv"] += row["macs"] - dw_macs
        drop_macs["GhostConv"] += row["macs"] - gh_macs

    projection["totals"] = {
        unit: {
            "params": audit["total_params"] - drop_params[unit],
            "gmacs": (audit["total_macs"] - drop_macs[unit]) / 1e9,
            "params_drop_percent": 100.0 * drop_params[unit] / audit["total_params"],
            "gmacs_drop_percent": 100.0 * drop_macs[unit] / audit["total_macs"],
        }
        for unit in ("DWConv", "GhostConv")
    }
    audit["projection"] = projection
    audit["generated_at"] = datetime.now(timezone.utc).isoformat()
    audit["label"] = args.label

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    lines = [
        f"# Layer cost audit — {args.label}",
        "",
        f"- Model YAML: `{args.model_yaml}`",
        f"- Input: {args.imgsz}x{args.imgsz}",
        f"- Total params: {audit['total_params']:,}",
        f"- Total MACs (conv+linear): {audit['total_macs'] / 1e9:.3f} G",
        f"- Ultralytics reference: params={audit['ultralytics_info']['n_params']:,}, "
        f"GFLOPs={audit['ultralytics_info']['gflops']:.3f}",
        "",
        "## Per-layer table",
        "",
        "| layer | module | path | out C | out HxW | params | MACs (M) | MACs share |",
        "|---:|---|---|---:|---|---:|---:|---:|",
    ]
    for row in audit["layers"]:
        hw = f"{row['out_shape'][-2]}x{row['out_shape'][-1]}" if row["out_shape"] else "-"
        out_c = row["out_channels"] if row["out_channels"] is not None else "-"
        lines.append(
            f"| {row['layer']} | {row['module']} | {row['path']} | {out_c} | {hw} | "
            f"{row['params']:,} | {row['macs'] / 1e6:.2f} | {100 * row['macs_share']:.1f}% |"
        )

    lines += [
        "",
        "## Unit-swap projection on high-level plain Conv layers",
        "",
        "| layer | path | C1->C2 | out HxW | orig params | DWConv params | GhostConv params | orig MACs (M) | DWConv MACs (M) | GhostConv MACs (M) |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for pos in projection["candidate_positions"]:
        lines.append(
            f"| {pos['layer']} | {pos['path']} | {pos['in_channels']}->{pos['out_channels']} | "
            f"{pos['out_shape'][-2]}x{pos['out_shape'][-1]} | {pos['original']['params']:,} | "
            f"{pos['dwconv']['params']:,} | {pos['ghostconv']['params']:,} | "
            f"{pos['original']['macs'] / 1e6:.2f} | {pos['dwconv']['macs'] / 1e6:.2f} | {pos['ghostconv']['macs'] / 1e6:.2f} |"
        )
    lines += [
        "",
        "| unit | projected params | projected GMACs | params drop | GMACs drop | Stage20 gate (params >=12%, GMACs >=8%) |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for unit, totals in projection["totals"].items():
        gate = "PASS" if totals["params_drop_percent"] >= 12 and totals["gmacs_drop_percent"] >= 8 else "FAIL"
        lines.append(
            f"| {unit} | {totals['params']:,} | {totals['gmacs']:.3f} | "
            f"-{totals['params_drop_percent']:.1f}% | -{totals['gmacs_drop_percent']:.1f}% | {gate} |"
        )
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")


if __name__ == "__main__":
    main()
