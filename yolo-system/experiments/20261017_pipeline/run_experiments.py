"""Reproducible HTTP comparison and 30 minute on-device sustained test."""
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
import traceback
from urllib.request import Request, urlopen

ROOT = Path('/home/<user>/projects/yolo-system')
OUT = Path(__file__).parent / time.strftime('run_%Y%m%d_%H%M%S')
OUT.mkdir()
BASE = 'http://127.0.0.1:8078'
PID = int(subprocess.check_output(['systemctl','--user','show','yolo-workbench','--property=MainPID','--value']))


def api(path, post=False):
    with urlopen(Request(BASE+path, data=b'' if post else None), timeout=15) as r:
        return json.load(r)


def sample(job):
    record = dict(time=time.time(), task=job['id'], frames=job['frames'], fps=job.get('fps'),
                  telemetry=api('/api/health')['telemetry'])
    status = Path(f'/proc/{PID}/status').read_text()
    record['rss_mb'] = int(next(x for x in status.splitlines() if x.startswith('VmRSS:')).split()[1])/1024
    rails = {}
    for label in Path('/sys/bus/i2c/drivers/ina3221').glob('*/hwmon/hwmon*/in*_label'):
        channel = label.stem.split('_')[0][2:]
        current = label.with_name(f'curr{channel}_input')
        voltage = label.with_name(f'in{channel}_input')
        if current.exists() and voltage.exists():
            rails[label.read_text().strip()] = int(voltage.read_text())*int(current.read_text())/1e6
    record['power_w'] = rails
    return record


def run(model, mode, duration=0):
    key = api(f'/api/jobs?kind=demo-video&model={model}&pipeline={mode}&duration_s={duration}&confidence=0.25', True)['id']
    print(f'START {model} {mode} {duration}s {key}', flush=True)
    with (OUT / f'{key}_telemetry.jsonl').open('w') as stream:
        while True:
            job = next(j for j in api('/api/jobs') if j['id']==key)
            stream.write(json.dumps(sample(job))+'\n')
            stream.flush()
            if job['status'] not in ('queued','running'):
                break
            time.sleep(1)
    assert job['status']=='completed', job
    (OUT / f'{key}.json').write_text(json.dumps(job,ensure_ascii=False,indent=2))
    print(f'DONE {key}: frames={job["frames"]}, fps={job["fps"]}, stages={job["stages_mean_ms"]}',flush=True)
    return job


def persist(data):
    (OUT / 'summary.json').write_text(json.dumps(data,ensure_ascii=False,indent=2))


if __name__=='__main__':
    data = dict(status='running', started_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), short=[], controls=[])
    try:
        assert api('/api/health')['active'] is None
        environment=dict(uname=list(os.uname()), service_pid=PID, source_hashes={})
        for name in ['server.py','video_pipeline.py','index.html']:
            environment['source_hashes'][name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        for key,path in [('nvpmodel',['nvpmodel','-q']),('jetson_clocks',['jetson_clocks','--show']),('processes',['ps','-eo','pid,comm,args'])]:
            p=subprocess.run(path,capture_output=True,text=True)
            environment[key]=p.stdout+p.stderr
        (OUT/'environment.json').write_text(json.dumps(environment,indent=2))
        persist(data)
        for pair in range(5):
            modes=['baseline','optimized'] if pair%2==0 else ['optimized','baseline']
            for mode in modes:
                data['short'].append(run('rfb',mode))
                persist(data)
        hashes={j['detections_sha256'] for j in data['short']}
        assert len(hashes)==1, 'Detection outputs changed across modes/repetitions'
        for model in ['l0','k1']:
            for _ in range(3):
                data['controls'].append(run(model,'optimized'))
                persist(data)
        data['comparison']={mode:dict(fps_median=statistics.median(j['fps'] for j in data['short'] if j['pipeline']==mode),
                                     runs=5, detection_sequence_equal=True) for mode in ['baseline','optimized']}
        persist(data)
        raw=(OUT/'tegrastats_sustained.log').open('w')
        monitor=subprocess.Popen(['tegrastats','--interval','1000'],stdout=raw,stderr=subprocess.STDOUT)
        try:
            data['sustained']=run('rfb','optimized',1800)
        finally:
            monitor.terminate()
            monitor.wait(timeout=10)
            raw.close()
        sustained=data['sustained']
        assert sustained['elapsed_s']>=1800 and sustained['loop_count']>1
        # Verify actual exported files, rather than trusting metadata alone.
        import cv2
        recorded=0
        for name in sustained['recording_files']:
            cap=cv2.VideoCapture(str(ROOT/'runtime'/sustained['id']/name))
            assert cap.isOpened(), name
            while True:
                ok,_=cap.read()
                if not ok:break
                recorded+=1
            cap.release()
        with (ROOT/'runtime'/sustained['id']/'frames.jsonl').open() as log:
            logged=sum(1 for _ in log)
        assert recorded==logged==sustained['frames'],(recorded,logged,sustained['frames'])
        data.update(status='PASS',recorded_frames_verified=recorded,logged_frames_verified=logged,
                    finished_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    except Exception:
        data.update(status='FAIL',error=traceback.format_exc())
        traceback.print_exc()
    finally:
        persist(data)
        print(f'FINAL {data["status"]} {OUT}',flush=True)
