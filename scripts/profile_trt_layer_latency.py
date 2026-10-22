"""TensorRT per-layer GPU-time profiling for a strongly-typed FP16 engine (P20).

Attaches a default IProfiler (its C++ implementation prints per-layer times to
stdout), redirects fd 1 to a temp file, runs repeated batch-1 forwards, then
parses and aggregates the printed table by top-level module prefix
(/model.<i>/..., Detect decode kernels, reformat copies). Measurement-only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LINE_RE = re.compile(r"^(.+?): ([0-9.eE+-]+)ms$")
MODULE_RE = re.compile(r"model[./](\d+)")
TOP_MODULES = (
    "Conv,Conv,C3k2,Conv,C3k2,Conv*,C3k2,Conv*,C3k2,SPPF,C2PSA,Upsample,Concat,C3k2,"
    "Upsample,Concat,C3k2,Upsample,ACFRConcat,C3k2,Conv,Concat,C3k2,Conv*,Concat,C3k2,Conv*,Concat,C3k2,Detect"
).split(",")


def build_engine(onnx: Path, cache: Path, logger, trt):
    if cache.is_file():
        return cache.read_bytes()
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx.read_bytes()):
        raise RuntimeError(f"ONNX parse failed: {onnx}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    profile = builder.create_optimization_profile()
    profile.set_shape("images", (1, 3, 640, 640), (1, 3, 640, 640), (1, 3, 640, 640))
    config.add_optimization_profile(profile)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("engine build failed")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(bytes(serialized))
    return bytes(serialized)


def parse_and_aggregate(text: str, runs: int) -> list[dict]:
    buckets: dict[str, dict] = defaultdict(lambda: {"total_ms": 0.0, "count": 0})
    for line in text.splitlines():
        match = LINE_RE.match(line.strip())
        if not match:
            continue
        name, value = match.group(1), float(match.group(2))
        module = MODULE_RE.search(name)
        if module:
            key = f"model.{module.group(1)}"
        elif name.startswith("__myl_"):
            key = "detect-decode"  # end2one decode kernels exported under Detect
        elif name.startswith("Reformatting"):
            inner = MODULE_RE.search(name.split(" to ", 1)[-1])
            key = f"model.{inner.group(1)}" if inner else "reformat-other"
        else:
            key = "other"
        entry = buckets[key]
        entry["total_ms"] += value
        entry["count"] += 1
    rows = []
    for key, entry in sorted(buckets.items(), key=lambda kv: int(kv[0].split(".")[1]) if kv[0].startswith("model.") else 99):
        per_run = entry["total_ms"] / runs
        rows.append(
            {
                "key": key,
                "top_level": int(key.split(".")[1]) if key.startswith("model.") else None,
                "module": TOP_MODULES[int(key.split(".")[1])] if key.startswith("model.") else key,
                "gpu_ms_per_run": round(per_run, 4),
                "reported_layer_runs": entry["count"],
            }
        )
    total = sum(r["gpu_ms_per_run"] for r in rows)
    for r in rows:
        r["share_percent"] = round(100.0 * r["gpu_ms_per_run"] / total, 2) if total else 0.0
    rows.sort(key=lambda r: -r["gpu_ms_per_run"])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=Path, help="existing strongly-typed FP16 engine")
    parser.add_argument("--onnx", type=Path, help="fp16 ONNX to build from if no engine")
    parser.add_argument("--engine-cache", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--device", type=str, default="1")
    parser.add_argument("--label", type=str, default="L0")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    import numpy as np
    import pycuda.driver as cuda
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.ERROR)
    if args.engine:
        serialized = args.engine.read_bytes()
    elif args.onnx:
        cache = args.engine_cache or (args.onnx.parent / f"{args.onnx.stem}_b1_prof.engine")
        serialized = build_engine(args.onnx, cache, logger, trt)
    else:
        raise SystemExit("need --engine or --onnx")

    cuda.init()
    ctx = cuda.Device(0).make_context()
    captured = tempfile.NamedTemporaryFile(mode="w+", suffix=".txt", delete=False)
    try:
        engine = trt.Runtime(logger).deserialize_cuda_engine(serialized)
        context = engine.create_execution_context()
        context.set_input_shape("images", (1, 3, 640, 640))
        inp = cuda.mem_alloc(1 * 3 * 640 * 640 * 2)
        outputs = {}
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                shape = tuple(context.get_tensor_shape(name))
                dtype = engine.get_tensor_dtype(name)
                itemsize = 2 if dtype == trt.DataType.HALF else 4
                ptr = cuda.mem_alloc(int(np.prod(shape)) * itemsize)
                outputs[name] = ptr
                context.set_tensor_address(name, int(ptr))
        context.set_tensor_address("images", int(inp))
        stream = cuda.Stream()
        sample = np.ascontiguousarray(np.random.default_rng(0).random((1, 3, 640, 640)).astype(np.float16))

        context.profiler = trt.Profiler()  # default C++ printer writes to fd 1

        def forward() -> None:
            cuda.memcpy_htod_async(inp, sample, stream)
            context.execute_async_v3(int(stream.handle))

        for _ in range(args.warmup):
            forward()
        stream.synchronize()

        sys.stdout.flush()
        saved = os.dup(1)
        os.dup2(captured.fileno(), 1)
        try:
            for _ in range(args.runs):
                forward()
            stream.synchronize()
        finally:
            sys.stdout.flush()
            os.dup2(saved, 1)
            os.close(saved)

        captured.flush()
        rows = parse_and_aggregate(Path(captured.name).read_text(), args.runs)
        total = sum(r["gpu_ms_per_run"] for r in rows)
        detect_rows = [r for r in rows if r["module"] == "Detect" or r["key"] == "detect-decode"]
        detect_share = round(sum(r["share_percent"] for r in detect_rows), 2)
        result = {
            "protocol": {
                "label": args.label,
                "engine": str(args.engine or args.onnx),
                "batch": 1,
                "imgsz": 640,
                "runs": args.runs,
                "device": f"physical GPU{args.device}",
                "tensorrt_version": trt.__version__,
                "note": "per-layer GPU time via default IProfiler stdout table; per-run averages; Detect decode kernels aggregated as detect-decode",
            },
            "profiled_layers_ms_per_run": round(total, 4),
            "detect_head_share_percent_incl_decode": detect_share,
            "per_top_level": rows,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"profiled total {total:.3f} ms/run; Detect(+decode) share {detect_share}%")
        for r in rows[:15]:
            print(f"  {r['key']:<16} {r['module']:<12} {r['gpu_ms_per_run']:>8.4f} ms  {r['share_percent']:>5.1f}%")
        print(f"wrote {args.output}")
    finally:
        ctx.pop()
        os.unlink(captured.name)


if __name__ == "__main__":
    main()
