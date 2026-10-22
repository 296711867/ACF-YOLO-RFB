"""TensorRT engine session, preprocessing and YOLO26 end-to-end postprocessing.

Shared by every Jetson-side benchmark / evaluation script in this package.
The engine protocol mirrors the frozen RTX-server protocol in
scripts/benchmark_tensorrt_fp16.py of the research repo: letterbox to 640 with
pad 114, BGR->RGB, /255, CHW, float32 host array cast to the engine input
dtype, pycuda stream + CUDA events for timing.

The deployed graphs are YOLO26 end-to-end (NMS-free): a single output tensor
"output0" of shape [batch, 300, 6] holding (x1, y1, x2, y2, score, class) in
letterboxed 640x640 pixel coordinates. Postprocessing is therefore just a
coordinate un-letterbox plus a confidence filter - no NMS is needed.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]

_ACTIVE_CONTEXT = None


def ensure_cuda_context():
    """Attach the primary CUDA context once per process (pycuda)."""
    global _ACTIVE_CONTEXT
    if _ACTIVE_CONTEXT is None:
        import pycuda.driver as cuda

        cuda.init()
        dev = cuda.Device(0)
        if hasattr(dev, "retain_context"):
            _ACTIVE_CONTEXT = dev.retain_context()
        else:
            # older pycuda without Device.retain_context: autoinit pushes a
            # context on this thread and registers proper atexit cleanup
            import pycuda.autoinit  # noqa: F401

            _ACTIVE_CONTEXT = pycuda.autoinit.context
    return _ACTIVE_CONTEXT


class TrtEngine:
    """Deserialize a TensorRT engine and run synchronous or timed forwards."""

    def __init__(self, engine_path: str | Path, batch: int = 1, imgsz: int = 640):
        import tensorrt as trt
        import pycuda.driver as cuda

        ensure_cuda_context()
        self.trt = trt
        self.cuda = cuda
        serialized = Path(engine_path).read_bytes()
        logger = trt.Logger(trt.Logger.WARNING)
        self.engine = trt.Runtime(logger).deserialize_cuda_engine(serialized)
        if self.engine is None:
            raise RuntimeError(f"engine deserialization failed: {engine_path}")
        self.context = self.engine.create_execution_context()

        self.input_name = None
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_name = name
                break
        if self.input_name is None:
            raise RuntimeError("engine has no input tensor")
        self.in_dtype = self.engine.get_tensor_dtype(self.input_name)
        static_shape = self.engine.get_tensor_shape(self.input_name)
        self.dynamic_input = any(isinstance(d, int) and d < 0 for d in static_shape) or any(
            not isinstance(d, int) for d in static_shape
        )
        self.context.set_input_shape(self.input_name, (batch, 3, imgsz, imgsz))

        itemsize = 2 if self.in_dtype == trt.DataType.HALF else 4
        self.inp = cuda.mem_alloc(batch * 3 * imgsz * imgsz * itemsize)
        self.out_allocs: dict[str, tuple] = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                shape = tuple(self.context.get_tensor_shape(name))
                dt = self.engine.get_tensor_dtype(name)
                out_itemsize = 2 if dt == trt.DataType.HALF else 4
                ptr = cuda.mem_alloc(int(np.prod(shape)) * out_itemsize)
                self.out_allocs[name] = (ptr, shape, dt)
                self.context.set_tensor_address(name, int(ptr))
        self.context.set_tensor_address(self.input_name, int(self.inp))
        self.stream = cuda.Stream()
        self.ev0 = cuda.Event()
        self.ev1 = cuda.Event()
        self.engine_path = str(engine_path)

    # -- forward variants ------------------------------------------------
    def _push(self, arr: np.ndarray) -> np.ndarray:
        if arr.dtype == np.float32 and self.in_dtype == self.trt.DataType.HALF:
            arr = arr.astype(np.float16)
        return np.ascontiguousarray(arr)

    def forward(self, arr: np.ndarray) -> None:
        """H2D copy + asynchronous forward; no device-to-host copy, no sync."""
        self.cuda.memcpy_htod_async(self.inp, self._push(arr), self.stream)
        self.context.execute_async_v3(int(self.stream.handle))

    def forward_timed_ms(self, arr: np.ndarray) -> float:
        """One forward (H2D inside the timed region) timed with CUDA events."""
        payload = self._push(arr)
        self.ev0.record(self.stream)
        self.cuda.memcpy_htod_async(self.inp, payload, self.stream)
        self.context.execute_async_v3(int(self.stream.handle))
        self.ev1.record(self.stream)
        self.ev1.synchronize()
        return self.ev0.time_till(self.ev1)

    def fetch(self) -> dict[str, np.ndarray]:
        """Copy outputs back to host (call after forward + stream sync)."""
        self.stream.synchronize()
        out = {}
        for name, (ptr, shape, dt) in self.out_allocs.items():
            dtype = np.float16 if dt == self.trt.DataType.HALF else np.float32
            arr = np.empty(int(np.prod(shape)), dtype=dtype)
            self.cuda.memcpy_dtoh(arr, int(ptr))
            out[name] = arr.reshape(shape)
        return out

    def infer(self, arr: np.ndarray) -> dict[str, np.ndarray]:
        """Synchronous convenience forward returning host outputs."""
        self.forward(arr)
        return self.fetch()


# -- preprocessing --------------------------------------------------------
def letterbox(img_bgr: np.ndarray, size: int = 640) -> tuple[np.ndarray, tuple[float, int, int]]:
    """Ultralytics-style letterbox: keep aspect, center, pad 114.

    Returns (canvas_bgr, params) where params = (scale r, pad_top, pad_left)
    maps original pixels into the letterboxed canvas.
    """
    import cv2

    h, w = img_bgr.shape[:2]
    r = size / max(h, w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top : top + nh, left : left + nw] = resized
    return canvas, (r, top, left)


def preprocess(img_bgr: np.ndarray, imgsz: int = 640) -> tuple[np.ndarray, tuple[float, int, int]]:
    """letterbox -> RGB -> CHW -> float32/255 -> NCHW."""
    canvas, params = letterbox(img_bgr, imgsz)
    chw = canvas[:, :, ::-1].transpose(2, 0, 1)
    return (chw.astype(np.float32) / 255.0)[None], params


# -- postprocessing -------------------------------------------------------
def decode_detections(
    output: np.ndarray,
    params: tuple[float, int, int],
    orig_hw: tuple[int, int],
    conf: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map [.,300,6] end-to-end output back to original-image xyxy pixels.

    Returns (boxes_xyxy, scores, class_ids) filtered by conf, sorted by score
    descending. Pure numpy, no NMS (YOLO26 end-to-end graph).
    """
    out = np.asarray(output, dtype=np.float32).reshape(-1, 6)
    keep = out[:, 4] >= conf
    out = out[keep]
    if not len(out):
        empty = np.zeros((0, 4), dtype=np.float32)
        return empty, np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.int64)
    r, top, left = params
    h, w = orig_hw
    boxes = np.stack(
        [
            (out[:, 0] - left) / r,
            (out[:, 1] - top) / r,
            (out[:, 2] - left) / r,
            (out[:, 3] - top) / r,
        ],
        axis=1,
    )
    boxes[:, 0::2] = boxes[:, 0::2].clip(0, w)
    boxes[:, 1::2] = boxes[:, 1::2].clip(0, h)
    order = np.argsort(-out[:, 4], kind="stable")
    return boxes[order], out[order, 4], out[order, 5].astype(np.int64)


def load_input_image(path: str | Path):
    """imread + preprocess; returns (model_input, letterbox_params, orig_hw, bgr)."""
    import cv2

    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    arr, params = preprocess(img)
    return arr, params, img.shape[:2], img


def package_root() -> Path:
    return _PACKAGE_ROOT
