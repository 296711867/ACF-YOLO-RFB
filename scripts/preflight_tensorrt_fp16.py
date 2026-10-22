"""TensorRT FP16 engine precheck for lightweight candidates (P20).

Parses each ONNX and builds an FP16 engine, runs one forward and checks the
outputs are finite. Two build modes:

- strongly_typed (default): matches the frozen benchmark pipeline, requires an
  fp16 ONNX source; every op must have a true FP16 implementation.
- fp16_flag: non-strongly-typed network from an fp32 ONNX with kFP16 enabled;
  unsupported ops fall back to FP32 (standard mixed-precision deployment).

This is a parse/build gate, not a timing benchmark.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--onnx",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="repeatable label=path to an ONNX export",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="1")
    parser.add_argument("--engine-cache", type=Path, default=None)
    parser.add_argument("--mode", choices=("strongly_typed", "fp16_flag"), default="strongly_typed")
    return parser.parse_args()


def build_runner(onnx_path: Path, batch: int, imgsz: int, cache: Path | None, mode: str):
    """Build via the frozen strongly-typed recipe, or the mixed-precision flag recipe."""
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    if mode == "strongly_typed":
        from benchmark_tensorrt_fp16 import TrtRunner

        return TrtRunner(onnx_path, batch, imgsz, cache)

    import time

    import numpy as np
    import pycuda.driver as cuda
    import tensorrt as trt

    class FlagFp16Runner:
        def __init__(self) -> None:
            self.trt = trt
            self.cuda = cuda
            if cache and cache.is_file():
                serialized = cache.read_bytes()
            else:
                logger = trt.Logger(trt.Logger.WARNING)
                builder = trt.Builder(logger)
                network = builder.create_network()
                parser = trt.OnnxParser(network, logger)
                if not parser.parse(onnx_path.read_bytes()):
                    errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
                    raise RuntimeError(f"ONNX parse failed: {onnx_path}: {errors}")
                config = builder.create_builder_config()
                config.set_flag(trt.BuilderFlag.FP16)
                config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
                profile = builder.create_optimization_profile()
                profile.set_shape("images", (1, 3, imgsz, imgsz), (batch, 3, imgsz, imgsz), (batch, 3, imgsz, imgsz))
                config.add_optimization_profile(profile)
                t0 = time.perf_counter()
                serialized = builder.build_serialized_network(network, config)
                if serialized is None:
                    raise RuntimeError(f"Engine build failed: {onnx_path}")
                print(f"[trt] built {onnx_path.name} batch={batch} fp16_flag in {time.perf_counter()-t0:.1f}s")
                if cache:
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    cache.write_bytes(bytes(serialized))
            logger = trt.Logger(trt.Logger.WARNING)
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(serialized)
            self.context = self.engine.create_execution_context()
            self.context.set_input_shape("images", (batch, 3, imgsz, imgsz))
            self.input_name = "images"
            self.in_dtype = self.engine.get_tensor_dtype(self.input_name)
            nbytes = batch * 3 * imgsz * imgsz * (2 if self.in_dtype == trt.DataType.HALF else 4)
            self.inp = cuda.mem_alloc(nbytes)
            self.out_allocs: dict = {}
            for i in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(i)
                if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                    shape = tuple(self.context.get_tensor_shape(name))
                    dtype = self.engine.get_tensor_dtype(name)
                    itemsize = 2 if dtype == trt.DataType.HALF else 4
                    ptr = cuda.mem_alloc(int(np.prod(shape)) * itemsize)
                    self.out_allocs[name] = (ptr, shape, dtype)
                    self.context.set_tensor_address(name, int(ptr))
            self.context.set_tensor_address(self.input_name, int(self.inp))
            self.stream = cuda.Stream()

        def forward(self, arr: np.ndarray) -> None:
            if arr.dtype == np.float32 and self.in_dtype == trt.DataType.HALF:
                arr = arr.astype(np.float16)
            self.cuda.memcpy_htod_async(self.inp, np.ascontiguousarray(arr), self.stream)
            self.context.execute_async_v3(int(self.stream.handle))

    return FlagFp16Runner()


def main() -> None:
    args = parse_args()
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    import numpy as np
    import pycuda.driver as cuda
    import tensorrt as trt

    cuda.init()
    ctx = cuda.Device(0).make_context()
    results: dict = {
        "protocol": {
            "tensorrt_version": trt.__version__,
            "batch": args.batch,
            "imgsz": args.imgsz,
            "device": f"physical GPU{args.device}",
            "mode": args.mode,
            "network": (
                "strongly-typed FP16 (fp16 ONNX source)"
                if args.mode == "strongly_typed"
                else "builder kFP16 flag, fp32 ONNX source, per-op FP32 fallback allowed"
            ),
        },
        "candidates": {},
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        for spec in args.onnx:
            label, path_text = spec.split("=", 1)
            onnx_path = Path(path_text) if Path(path_text).is_absolute() else ROOT / path_text
            cache = args.engine_cache / f"{onnx_path.stem}_b{args.batch}_{args.mode}.engine" if args.engine_cache else None
            try:
                runner = build_runner(onnx_path, args.batch, args.imgsz, cache, args.mode)
                arr = np.random.rand(args.batch, 3, args.imgsz, args.imgsz).astype(np.float32)
                runner.forward(arr)
                runner.stream.synchronize()
                finite = True
                outputs = {}
                for name, (ptr, shape, dtype) in runner.out_allocs.items():
                    count = int(np.prod(shape))
                    host = np.empty(count, dtype=np.float16 if "HALF" in str(dtype) else np.float32)
                    cuda.memcpy_dtoh(host, ptr)
                    finite &= bool(np.isfinite(host.astype(np.float32)).all())
                    outputs[name] = {"shape": list(shape), "finite": True}
                results["candidates"][label] = {
                    "onnx": str(onnx_path),
                    "engine_built": True,
                    "parse_errors": 0,
                    "forward_finite": finite,
                    "outputs": outputs,
                    "status": "PASS" if finite else "FAIL_NON_FINITE",
                }
                print(f"[precheck] {label}: {'PASS' if finite else 'FAIL_NON_FINITE'}")
            except Exception as error:  # noqa: BLE001 - any build/parse failure is recorded as evidence
                results["candidates"][label] = {
                    "onnx": str(onnx_path),
                    "engine_built": False,
                    "error": repr(error)[:500],
                    "status": "FAIL",
                }
                print(f"[precheck] {label}: FAIL ({error!r})")
    finally:
        ctx.pop()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

