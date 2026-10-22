"""Live integration check: python smoke_test.py [http://127.0.0.1:8078]."""
import json
from pathlib import Path
import sys
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

BASE = sys.argv[1] if len(sys.argv) > 1 else 'http://127.0.0.1:8078'


def call(path, body=None, headers=None, expected=200):
    try:
        response = urlopen(Request(BASE + path, data=body, headers=headers or {}), timeout=15)
    except HTTPError as exc:
        response = exc
    assert response.code == expected, (path, response.code, response.read())
    payload = response.read()
    return json.loads(payload) if response.headers.get_content_type() == 'application/json' else payload


def wait(key):
    for _ in range(120):
        job = next(j for j in call('/api/jobs') if j['id'] == key)
        if job['status'] not in ('queued', 'running'):
            return job
        time.sleep(.5)
    raise AssertionError('Task did not finish in 60 seconds')


health = call('/api/health')
assert health['ok'] and all(m['available'] for m in health['models'])
assert health['active'] is None, 'Run this check while workbench is idle'
call('/api/jobs?kind=demo-image&confidence=nan', b'', expected=400)
call('/api/jobs?kind=demo-image', b'', {'Origin': 'http://another-site.example'}, expected=403)
call('/artifacts/../history.sqlite3', expected=400)
key = call('/api/jobs?kind=demo-image&model=compare', b'', expected=202)['id']
job = wait(key)
assert job['status'] == 'completed', job
assert set(job['results']) == {'rfb', 'l0', 'k1'}
for r in job['results'].values():
    assert r['count'] > 0 and len(r['engine_sha256']) == 64
    assert call(r['image_url']).startswith(b'\xff\xd8')
image_id = key
# Upload the returned JPEG through the same API the browser uses.
jpg = call(job['results']['rfb']['image_url'])
key = call('/api/jobs?kind=image&model=rfb&name=test.jpg', jpg, expected=202)['id']
assert wait(key)['status'] == 'completed'
key = call('/api/jobs?kind=image', b'not an image', expected=202)['id']
assert wait(key)['status'] == 'failed'
# A short, genuine video exercises upload, decode, inference, JSONL and encoder.
import cv2
source = Path('/home/<user>/projects/yolo_jetson/jetson_deploy/data/car.mp4')
cap = cv2.VideoCapture(str(source))
ok, frame = cap.read()
assert ok
clip = Path('/tmp/yolo-workbench-smoke.avi')
h, w = frame.shape[:2]
writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*'MJPG'), 30, (w, h))
assert writer.isOpened()
for _ in range(12):
    writer.write(frame)
    ok, frame = cap.read()
    assert ok
writer.release()
cap.release()
key = call('/api/jobs?kind=video&model=rfb', clip.read_bytes(), expected=202)['id']
video = wait(key)
assert video['status'] == 'completed' and video['frames'] == 12, video
assert len(call(video['frames_url']).splitlines()) == 12
assert call(video['video_url']).startswith(b'RIFF')
key = call('/api/jobs?kind=demo-video&model=l0', b'', expected=202)['id']
call('/api/jobs?kind=demo-image', b'', expected=409)
time.sleep(1)
call('/api/stop', b'')
stopped = wait(key)
assert stopped['status'] == 'stopped' and stopped['frames'] > 0, stopped
assert b'confidence,status,frames' in call('/api/history.csv')
report = dict(checked_at=time.strftime('%Y-%m-%d %H:%M:%S %z'),
              checks=['health and models', 'invalid threshold', 'cross-origin rejection', 'path traversal rejection',
                      'three-model real inference', 'image upload', 'invalid image failure record',
                      '12-frame video upload and export', 'busy conflict', 'stop and save', 'CSV export'],
              image_job=image_id, video_job=video['id'], stop_job=stopped['id'],
              video_frames=video['frames'], stop_frames=stopped['frames'], status='PASS')
Path(__file__).with_name('ACCEPTANCE.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(report, ensure_ascii=False, indent=2))
