"""TensorRT FP16 inference benchmark for ACR vs ACFR (pure engine forward).

Companion to benchmark_inference_speed.py (PyTorch FP32 end-to-end). This
script builds FP16 engines from ONNX exports and times pure engine forwards
(inputs resident on GPU) with CUDA events, using the same image list and
warmup/timed/repeat protocol so the ACR-vs-ACFR relative delta is directly
comparable within each precision tier. Absolute values are NOT comparable
across tiers: PyTorch numbers include preprocessing and Python overhead.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--acr", type=Path, required=True, help="ACR ONNX")
    parser.add_argument("--acfr", type=Path, required=True, help="ACFR ONNX")
    parser.add_argument("--image-list", type=Path, required=True)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engine-cache", type=Path, default=None)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--warmup-images", type=int, default=48)
    parser.add_argument("--timed-images", type=int, default=52)
    parser.add_argument("--repeat", type=int, default=3)
    return parser.parse_args()


def letterbox(img: np.ndarray, size: int) -> np.ndarray:
    import cv2

    h, w = img.shape[:2]
    r = size / max(h, w)
    nh, nw = int(round(h * r)), int(round(w * r))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top : top + nh, left : left + nw] = img
    return canvas


def load_batches(args: argparse.Namespace) -> list[np.ndarray]:
    import cv2

    paths = [line.strip() for line in args.image_list.read_text().splitlines() if line.strip()]
    images = [args.images_root / p for p in paths]
    timed = images[args.warmup_images : args.warmup_images + args.timed_images]
    batches = []
    for i in range(0, len(timed) - args.batch + 1, args.batch):
        chunk = timed[i : i + args.batch]
        arr = np.stack(
            [
                letterbox(cv2.imread(str(p)), args.imgsz)[:, :, ::-1].transpose(2, 0, 1)[None]
                for p in chunk
            ]
        )
        batches.append(np.ascontiguousarray(arr.astype(np.float32) / 255.0))
    return batches


class TrtRunner:
    """Build (or load cached) FP16 engine and run pure-forward timing."""

    def __init__(self, onnx: Path, batch: int, imgsz: int, cache: Path | None):
        import pycuda.driver as cuda
        import tensorrt as trt

        self.trt = trt
        self.cuda = cuda
        self.batch = batch
        if cache and cache.is_file():
            serialized = cache.read_bytes()
        else:
            logger = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(logger)
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
            )  # TRT11: precision inherited from fp16 ONNX, builder FP16 flag removed
            parser = trt.OnnxParser(network, logger)
            if not parser.parse(onnx.read_bytes()):
                for i in range(parser.num_errors):
                    print(parser.get_error(i))
                raise RuntimeError(f"ONNX parse failed: {onnx}")
            config = builder.create_builder_config()
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
            profile = builder.create_optimization_profile()
            shape = (batch, 3, imgsz, imgsz)
            profile.set_shape("images", (1, 3, imgsz, imgsz), shape, shape)
            config.add_optimization_profile(profile)
            t0 = time.perf_counter()
            serialized = builder.build_serialized_network(network, config)
            if serialized is None:
                raise RuntimeError(f"Engine build failed: {onnx}")
            print(f"[trt] built {onnx.name} batch={batch} in {time.perf_counter()-t0:.1f}s")
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
                dt = self.engine.get_tensor_dtype(name)
                itemsize = 2 if dt == trt.DataType.HALF else 4
                ptr = cuda.mem_alloc(int(np.prod(shape)) * itemsize)
                self.out_allocs[name] = (ptr, shape, dt)
                self.context.set_tensor_address(name, int(ptr))
        self.context.set_tensor_address(self.input_name, int(self.inp))
        self.stream = cuda.Stream()
        self.ev0 = cuda.Event()
        self.ev1 = cuda.Event()

    def forward(self, arr: np.ndarray) -> None:
        import tensorrt as trt

        if arr.dtype == np.float32 and self.in_dtype == trt.DataType.HALF:
            arr = arr.astype(np.float16)
        self.cuda.memcpy_htod_async(self.inp, np.ascontiguousarray(arr), self.stream)
        self.context.execute_async_v3(int(self.stream.handle))

    def time_batches(self, batches: list[np.ndarray], warmup_forwards: int, repeat: int) -> list[float]:
        for _ in range(warmup_forwards):
            self.forward(batches[0])
        self.stream.synchronize()
        runs = []
        for _ in range(repeat):
            self.ev0.record(self.stream)
            for b in batches:
                self.forward(b)
            self.ev1.record(self.stream)
            self.ev1.synchronize()
            runs.append(self.ev0.time_till(self.ev1))
        return runs


def main() -> None:
    args = parse_args()
    import os

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.device)
    import tensorrt as trt

    import pycuda.driver as cuda

    cuda.init()
    ctx = cuda.Device(0).make_context()

    batches = load_batches(args)
    n_images = len(batches) * args.batch
    results: dict = {
        "protocol": {
            "device": args.device,
            "batch": args.batch,
            "fp16": True,
            "tensorrt_version": trt.__version__,
            "timing": "pure engine forward (inputs copied H2D inside timed region), CUDA events",
            "warmup_forwards": max(1, args.warmup_images // args.batch),
            "timed_images": n_images,
            "repeat": args.repeat,
            "image_list": str(args.image_list),
            "note": "absolute ms not comparable with PyTorch FP32 end-to-end json; compare relative delta within tier",
        }
    }
    for label, onnx in (("acr", args.acr), ("acfr", args.acfr)):
        cache = args.engine_cache / f"{onnx.stem}_b{args.batch}.engine" if args.engine_cache else None
        runner = TrtRunner(onnx, args.batch, args.imgsz, cache)
        warmup_forwards = max(1, args.warmup_images // args.batch)
        ms_runs = runner.time_batches(batches, warmup_forwards, args.repeat)
        per_image = [round(ms / n_images, 4) for ms in ms_runs]
        results[label] = {
            "onnx": str(onnx),
            "ms_per_run": [round(ms, 3) for ms in ms_runs],
            "ms_per_image": per_image,
            "ms_per_image_mean": round(sum(per_image) / len(per_image), 4),
        }
    delta = results["acfr"]["ms_per_image_mean"] / results["acr"]["ms_per_image_mean"] - 1
    results["acfr_vs_acr_ms_delta_percent"] = round(delta * 100, 2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    ctx.pop()
    print(json.dumps({k: results[k] for k in ("acr", "acfr", "acfr_vs_acr_ms_delta_percent")}, indent=2))


if __name__ == "__main__":
    main()
