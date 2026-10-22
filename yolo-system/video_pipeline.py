"""Recording and browser preview shared by production and comparison runs."""
import time
from pathlib import Path

import cv2


class VideoOutput:
    def __init__(self, folder, fps, mode='optimized'):
        self.folder, self.fps, self.mode = Path(folder), fps, mode
        self.writer = None
        self.frames = 0
        self.latest = None
        self.preview_at = 0
        self.files = []

    def write(self, frame):
        started = time.perf_counter()
        # AVI segments avoid oversized single-file exports during sustained tests.
        if self.frames % 4500 == 0:
            self.close()
            name = 'annotated.avi' if not self.files else f'annotated_{len(self.files):03}.avi'
            h, w = frame.shape[:2]
            codec = 'MJPG' if self.mode == 'baseline' else 'mp4v'
            self.writer = cv2.VideoWriter(str(self.folder / name), cv2.VideoWriter_fourcc(*codec), self.fps, (w, h))
            if not self.writer.isOpened():
                raise RuntimeError(f'Cannot open video writer: {codec}')
            self.files.append(name)
        self.writer.write(frame)
        recorded = time.perf_counter()
        previewed = False
        if self.mode == 'baseline' or recorded - self.preview_at >= .1:
            preview = frame
            if self.mode != 'baseline' and frame.shape[1] > 960:
                h, w = frame.shape[:2]
                preview = cv2.resize(frame, (960, round(h * 960 / w)), interpolation=cv2.INTER_AREA)
            ok, jpg = cv2.imencode('.jpg', preview, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                raise RuntimeError('JPEG encoding failed')
            self.latest = jpg.tobytes()
            self.preview_at = recorded
            previewed = True
        self.frames += 1
        return {'record_ms': (recorded - started) * 1000,
                'preview_ms': (time.perf_counter() - recorded) * 1000, 'preview_encoded': previewed}

    def close(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
