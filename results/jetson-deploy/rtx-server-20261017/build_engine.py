"""Build TensorRT engines from the frozen ONNX exports (run on the Jetson).

Modes:
  fp16            explicit-precision network + BuilderFlag.FP16 (thesis FP16
                  deployment; float32 IO, FP16 kernels)
  fp32            no precision flag (reference engine for the numeric
                  consistency check and for bounding the FP16 error)
  strongly_typed  strongly-typed network, identical to the mode used for the
                  RTX-side §11.3 engines (note: the exported ONNX is all
                  float32, so strongly-typed engines execute at FP32
                  precision; kept for cross-machine comparability)

Engines are hardware-specific: always build on the device that will run them.
An RTX engine file must never be copied to Jetson.

Example:
  python3 scripts/build_engine.py --onnx models/l0_ep200.onnx --mode fp16
  python3 scripts/build_engine.py --onnx models/l0_ep200.onnx --mode fp32
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--mode", choices=("fp16", "fp32", "strongly_typed"), default="fp16")
    parser.add_argument("--engine-out", type=Path, default=None)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workspace-gb", type=float, default=2.0)
    parser.add_argument("--verify", action="store_true", help="run one zero-input forward and check outputs")
    parser.add_argument("--record-json", type=Path, default=None)
    return parser.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    args = parse_args()
    import tensorrt as trt
    from trt_runner import TrtEngine, ensure_cuda_context

    onnx_path = args.onnx if args.onnx.is_absolute() else PACKAGE_ROOT / args.onnx
    if not onnx_path.is_file():
        raise SystemExit(f"ONNX not found: {onnx_path}")
    engine_out = args.engine_out or (
        PACKAGE_ROOT / "engines" / f"{onnx_path.stem}_{args.mode}_b{args.batch}.engine"
    )
    engine_out.parent.mkdir(parents=True, exist_ok=True)

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    if args.mode == "strongly_typed":
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    else:
        # TRT 8.x ONNX parser requires EXPLICIT_BATCH; the flag was removed in
        # TRT 10 where explicit batch is the only mode, so probe before use.
        flags = 0
        explicit_batch = getattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH", None)
        if explicit_batch is not None:
            flags |= 1 << int(explicit_batch)
        network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise SystemExit(f"ONNX parse failed: {onnx_path}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(args.workspace_gb * (1 << 30)))
    if args.mode == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)

    profile = builder.create_optimization_profile()
    shape = (args.batch, 3, args.imgsz, args.imgsz)
    profile.set_shape("images", (1, 3, args.imgsz, args.imgsz), shape, shape)
    config.add_optimization_profile(profile)

    t0 = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise SystemExit(f"engine build failed: {onnx_path}")
    build_seconds = time.perf_counter() - t0
    engine_out.write_bytes(bytes(serialized))
    print(f"[build] {onnx_path.name} mode={args.mode} batch={args.batch} -> {engine_out}")
    print(f"[build] took {build_seconds:.1f}s, engine size {engine_out.stat().st_size / 1e6:.1f} MB")

    record = {
        "onnx": str(onnx_path),
        "onnx_sha256": sha256(onnx_path),
        "mode": args.mode,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "workspace_gb": args.workspace_gb,
        "engine": str(engine_out),
        "engine_sha256": sha256(engine_out),
        "engine_bytes": engine_out.stat().st_size,
        "tensorrt_version": trt.__version__,
        "build_seconds": round(build_seconds, 1),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }

    if args.verify:
        import numpy as np

        ensure_cuda_context()
        engine = TrtEngine(engine_out, batch=args.batch, imgsz=args.imgsz)
        arr = np.zeros((args.batch, 3, args.imgsz, args.imgsz), dtype=np.float32)
        out = engine.infer(arr)
        finite = True
        shapes = {}
        for name, tensor in out.items():
            shapes[name] = list(tensor.shape)
            finite = finite and bool(np.isfinite(tensor).all())
        expected = [args.batch, 300, 6]
        record["verify"] = {"output_shapes": shapes, "all_finite": finite}
        status = "PASS" if finite and list(shapes.values())[0] == expected else "FAIL"
        record["verify"]["status"] = status
        print(f"[verify] shapes={shapes} finite={finite} -> {status}")
        if status != "PASS":
            raise SystemExit(1)

    record_path = args.record_json or (
        PACKAGE_ROOT / "results" / f"engine_build_{onnx_path.stem}_{args.mode}_b{args.batch}.json"
    )
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"[record] {record_path}")


if __name__ == "__main__":
    main()
