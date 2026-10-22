"""LAN research workbench; reuse the frozen Jetson TensorRT pipeline."""
import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
import threading
import time
import traceback
from contextlib import closing
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit
from uuid import uuid4

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
DEPLOY = Path(os.environ.get('YOLO_DEPLOY_ROOT', '/home/<user>/projects/yolo_jetson/jetson_deploy'))
DATA = ROOT / 'runtime'
MODELS = {'rfb': 'ACF-YOLO-RFB', 'l0': 'ACF-YOLO', 'k1': 'ACF-YOLO-KD'}
NAMES = ['person', 'rider', 'car', 'truck', 'bus', 'train', 'motor', 'bike', 'traffic light', 'traffic sign']
POOL = ThreadPoolExecutor(max_workers=1)  # ponytail: one CUDA owner; queue multi-user workloads when needed.
LOCK = threading.Lock()
ACTIVE = None
STOP = threading.Event()
LATEST = None
ENGINES = {}


def db():
    return closing(sqlite3.connect(DATA / 'history.sqlite3', timeout=10))


def initialize():
    DATA.mkdir(exist_ok=True)
    with db() as con:
        con.execute('CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, record TEXT NOT NULL)')
        for key, raw in con.execute('SELECT id, record FROM jobs').fetchall():
            item = json.loads(raw)
            if item['status'] in ('queued', 'running'):
                item.update(status='interrupted', error='Service restarted; task did not complete.')
                con.execute('UPDATE jobs SET record=? WHERE id=?', (json.dumps(item), key))
        con.commit()


def save(item):
    with db() as con:
        con.execute('INSERT OR REPLACE INTO jobs VALUES (?,?)', (item['id'], json.dumps(item, ensure_ascii=False)))
        con.commit()
    (DATA / item['id'] / 'result.json').write_text(json.dumps(item, ensure_ascii=False, indent=2), encoding='utf-8')


def history():
    with db() as con:
        return [json.loads(r[0]) for r in con.execute('SELECT record FROM jobs ORDER BY rowid DESC LIMIT 100')]


def safe_artifact(path):
    target = (DATA / path).resolve()
    if not target.is_relative_to(DATA.resolve()) or not target.is_file():
        raise ValueError('Result file does not exist')
    if target.suffix not in ('.jpg', '.json', '.jsonl', '.avi'):
        raise ValueError('Unsupported result file')
    return target


def engine(model):
    if model not in ENGINES:
        sys.path.insert(0, str(DEPLOY / 'scripts'))
        from trt_runner import TrtEngine
        path = DEPLOY / 'engines' / f'{model}_ep200_fp16_b1.engine'
        session = TrtEngine(path)
        sample = np.zeros((1, 3, 640, 640), np.float32)
        for _ in range(10):
            session.infer(sample)
        ENGINES[model] = (session, hashlib.sha256(path.read_bytes()).hexdigest())
    return ENGINES[model]


def infer(frame, model, conf):
    session, digest = engine(model)
    from trt_runner import preprocess, decode_detections
    started = time.perf_counter()
    arr, params = preprocess(frame)
    gpu_ms = session.forward_timed_ms(arr)
    output = session.fetch()
    boxes, scores, classes = decode_detections(next(iter(output.values())), params, frame.shape[:2], conf)
    detections = []
    for box, score, cls in zip(boxes, scores, classes):
        area = float((box[2] - box[0]) * (box[3] - box[1]))
        small = area < 32 ** 2
        detections.append(dict(box=box.tolist(), confidence=float(score), class_id=int(cls),
                               label=NAMES[int(cls)], small=small))
        x1, y1, x2, y2 = map(int, box)
        color = (80, 215, 255) if small else (155, 220, 60)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f'{NAMES[int(cls)]} {score:.2f}', (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1)
    return frame, dict(detections=detections, count=len(detections), small_count=sum(d['small'] for d in detections),
                      gpu_ms=round(gpu_ms, 3), pipeline_ms=round((time.perf_counter() - started) * 1000, 3), engine_sha256=digest)


def run_job(job, source):
    global ACTIVE, LATEST
    cap = writer = None
    folder = DATA / job['id']
    try:
        job['status'] = 'running'
        if source is not None:
            with source.open('rb') as stream:
                job['source_sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
            job['source_file'] = source.name
        save(job)
        models = list(MODELS) if job['model'] == 'compare' else [job['model']]
        for model in models:
            engine(model)
        started = time.perf_counter()
        if job['kind'] == 'image':
            image = cv2.imread(str(source))
            if image is None:
                raise ValueError('无法解码图片，请上传 JPG 或 PNG')
            results = {}
            for model in models:
                annotated, result = infer(image.copy(), model, job['confidence'])
                if not cv2.imwrite(str(folder / f'{model}.jpg'), annotated):
                    raise RuntimeError('Cannot save annotated image')
                result['image_url'] = f"/artifacts/{job['id']}/{model}.jpg"
                results[model] = result
                LATEST = (folder / f'{model}.jpg').read_bytes()
            job.update(results=results, frames=1, elapsed_s=round(time.perf_counter() - started, 3))
            primary = results.get('rfb', next(iter(results.values())))
            job.update(count=primary['count'], small_count=primary['small_count'], gpu_mean_ms=primary['gpu_ms'])
        else:
            cap = cv2.VideoCapture(0 if job['kind'] == 'camera' else str(source))
            if not cap.isOpened():
                raise ValueError('无法打开输入源；摄像头需要 Jetson 已连接 /dev/video0')
            fps = cap.get(cv2.CAP_PROP_FPS)
            fps = fps if math.isfinite(fps) and fps > 0 else 30
            job['total_frames'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if job['kind'] != 'camera' else 0
            timings, gpu_timings, count, small = [], [], 0, 0
            with (folder / 'frames.jsonl').open('w', encoding='utf-8') as log:
                while not STOP.is_set() and job.get('frames', 0) < 9000:
                    tick = time.perf_counter()
                    ok, frame = cap.read()
                    if not ok:
                        break
                    annotated, result = infer(frame, job['model'], job['confidence'])
                    if writer is None:
                        h, w = frame.shape[:2]
                        writer = cv2.VideoWriter(str(folder / 'annotated.avi'), cv2.VideoWriter_fourcc(*'MJPG'), fps, (w, h))
                        if not writer.isOpened():
                            raise RuntimeError('Cannot open MJPEG video writer')
                    writer.write(annotated)
                    ok, jpg = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if not ok:
                        raise RuntimeError('JPEG encoding failed')
                    LATEST = jpg.tobytes()
                    frame_no = job.get('frames', 0) + 1
                    log.write(json.dumps(dict(frame=frame_no, **result)) + '\n')
                    timings.append((time.perf_counter() - tick) * 1000)
                    gpu_timings.append(result['gpu_ms'])
                    count += result['count']
                    small += result['small_count']
                    job.update(frames=frame_no, count=count, small_count=small, last=result,
                               elapsed_s=round(time.perf_counter() - started, 2),
                               fps=round(frame_no / (time.perf_counter() - started), 2))
                    if frame_no % 20 == 0:
                        save(job)
            if not timings:
                raise ValueError('输入源没有可读取的视频帧')
            (folder / 'preview.jpg').write_bytes(LATEST)
            job.update(p50_ms=round(float(np.median(timings)), 3), p95_ms=round(float(np.percentile(timings, 95)), 3),
                       gpu_mean_ms=round(float(np.mean(gpu_timings)), 3),
                       video_url=f"/artifacts/{job['id']}/annotated.avi", image_url=f"/artifacts/{job['id']}/preview.jpg",
                       frames_url=f"/artifacts/{job['id']}/frames.jsonl",
                       stopped=STOP.is_set(), truncated=job['frames'] >= 9000)
        job['status'] = 'stopped' if STOP.is_set() else 'completed'
    except Exception as exc:
        traceback.print_exc()
        job.update(status='failed', error=str(exc))
    finally:
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()
        job['finished_at'] = datetime.now(timezone.utc).isoformat()
        save(job)
        with LOCK:
            ACTIVE = None


def telemetry():
    thermals = []
    for zone in Path('/sys/class/thermal').glob('thermal_zone*'):
        try:
            thermals.append(dict(name=(zone / 'type').read_text().strip(), celsius=int((zone / 'temp').read_text()) / 1000))
        except OSError:
            continue
    mem = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    return dict(temperature=thermals, memory_used_gb=round((int(mem['MemTotal'].split()[0]) - int(mem['MemAvailable'].split()[0])) / 1048576, 2),
                memory_total_gb=round(int(mem['MemTotal'].split()[0]) / 1048576, 2), load=os.getloadavg(),
                camera_available=Path('/dev/video0').exists(), disk_free_gb=round(__import__('shutil').disk_usage(DATA).free / 1024**3, 1))


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def send(self, status, body, mime='application/json; charset=utf-8'):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(status)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        try:
            path = urlsplit(self.path).path
            if path == '/':
                return self.send(200, (ROOT / 'index.html').read_bytes(), 'text/html; charset=utf-8')
            if path == '/api/health':
                return self.send(200, dict(ok=True, models=[dict(id=k, name=v, available=(DEPLOY / 'engines' / f'{k}_ep200_fp16_b1.engine').exists()) for k, v in MODELS.items()],
                                           device=Path('/proc/device-tree/model').read_bytes().rstrip(b'\0').decode(), telemetry=telemetry(), active=ACTIVE))
            if path == '/api/jobs':
                return self.send(200, history())
            if path == '/api/latest.jpg':
                return self.send(200, LATEST, 'image/jpeg') if LATEST else self.send(404, dict(error='No frame yet'))
            if path == '/api/history.csv':
                out = io.StringIO()
                writer = csv.writer(out)
                writer.writerow(['id', 'created_at', 'kind', 'model', 'confidence', 'status', 'frames', 'fps', 'p50_ms', 'p95_ms', 'count', 'small_count'])
                for j in history():
                    writer.writerow([j.get(k, '') for k in ['id', 'created_at', 'kind', 'model', 'confidence', 'status', 'frames', 'fps', 'p50_ms', 'p95_ms', 'count', 'small_count']])
                return self.send(200, '\ufeff' + out.getvalue(), 'text/csv; charset=utf-8')
            if path.startswith('/artifacts/'):
                file = safe_artifact(path.removeprefix('/artifacts/'))
                mime = {'.jpg': 'image/jpeg', '.json': 'application/json', '.jsonl': 'application/x-ndjson', '.avi': 'video/x-msvideo'}[file.suffix]
                # Stream videos to avoid loading a full export into Jetson RAM.
                self.send_response(200)
                self.send_header('Content-Type', mime)
                self.send_header('Content-Length', str(file.stat().st_size))
                self.send_header('X-Content-Type-Options', 'nosniff')
                self.end_headers()
                with file.open('rb') as stream:
                    __import__('shutil').copyfileobj(stream, self.wfile)
                return
            self.send(404, dict(error='Unknown endpoint'))
        except (BrokenPipeError, ConnectionResetError):
            pass
        except (ValueError, OSError) as exc:
            self.send(400, dict(error=str(exc)))

    def do_POST(self):
        global ACTIVE, LATEST
        self.close_connection = True
        try:
            # Same-origin LAN console: prevent a third-party page from submitting jobs.
            origin = self.headers.get('Origin')
            if origin and origin != 'http://' + self.headers.get('Host', ''):
                return self.send(403, dict(error='Origin rejected'))
            path = urlsplit(self.path)
            if path.path == '/api/stop':
                STOP.set()
                return self.send(200, dict(stopping=ACTIVE is not None))
            if path.path != '/api/jobs':
                return self.send(404, dict(error='Unknown endpoint'))
            query = {k: v[0] for k, v in parse_qs(path.query).items()}
            kind, model = query.get('kind', 'image'), query.get('model', 'rfb')
            conf = float(query.get('confidence', '.25'))
            if kind not in ('image', 'demo-image', 'video', 'demo-video', 'camera') or model not in (*MODELS, 'compare') or not math.isfinite(conf) or not 0 < conf <= 1:
                raise ValueError('Invalid input type, model or confidence')
            if model == 'compare' and kind not in ('image', 'demo-image'):
                raise ValueError('三模型对比仅支持图片')
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 <= length <= 200 * 1024**2:
                return self.send(413, dict(error='文件上限 200 MB'))
            if kind in ('image', 'video') and not length:
                raise ValueError('请先选择文件')
            with LOCK:
                if ACTIVE is not None:
                    self.close_connection = True
                    return self.send(409, dict(error='已有检测任务，请等待或停止后重试'))
                key = datetime.now().strftime('%Y%m%d_%H%M%S_') + uuid4().hex[:8]
                folder = DATA / key
                folder.mkdir()
                ACTIVE = key
                STOP.clear()
                LATEST = None
                job = dict(id=key, created_at=datetime.now(timezone.utc).isoformat(), kind=kind.removeprefix('demo-'),
                           model=model, confidence=conf, status='queued', frames=0,
                           input_name=Path(query.get('name', kind)).name,
                           protocol=dict(imgsz=640, batch=1, precision='FP16', warmup=10, small_area='original image box area < 1024 px²',
                                         timing='GPU CUDA events include H2D; video pipeline includes read, preprocessing, GPU, D2H, decode, draw, JPEG, video encode; excludes HTTP/browser; not thesis benchmark protocol'))
                try:
                    if kind.startswith('demo-'):
                        source = DEPLOY / 'data' / 'car.mp4' if kind == 'demo-video' else next((DEPLOY / 'data/frozen_subset/images').glob('*.jpg'))
                    elif kind == 'camera':
                        source = None
                    else:
                        source = folder / ('input.jpg' if kind == 'image' else 'input.mp4')
                        self.connection.settimeout(60)
                        remaining = length
                        with source.open('wb') as out:
                            while remaining:
                                block = self.rfile.read(min(1024**2, remaining))
                                if not block:
                                    raise ValueError('上传中断')
                                out.write(block)
                                remaining -= len(block)
                    save(job)
                    POOL.submit(run_job, job, source)
                except Exception:
                    ACTIVE = None
                    raise
            self.send(202, dict(id=key))
        except (ValueError, OSError, StopIteration) as exc:
            self.send(400, dict(error=str(exc) or '演示素材不存在'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8078)
    args = parser.parse_args()
    initialize()
    print(f'YOLO workbench http://{args.host}:{args.port}', flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
