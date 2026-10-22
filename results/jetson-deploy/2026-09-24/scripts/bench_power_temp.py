"""Power / temperature / memory sampler wrapping any benchmark command.

Implements the measurement side of deployment-spec §16.2 items 5-7 by sampling
`tegrastats` while another command runs:

  python3 scripts/bench_power_temp.py --label l0_latency -- \
      python3 scripts/bench_latency.py --engine engines/l0_fp16_b1.engine ...

Parses both the Orin-family ("VDD_GPU_SOC", "VDD_IN", "GPU@x.xC") and the
Xavier/TX2-family ("POM_5V_GPU", "POM_5V_IN") tegrastats fields. On unified
memory Jetsons "peak memory" is reported as peak RAM in use (tegrastats) plus
the engine file size; there is no separate GPU memory pool.

If samples come back empty, re-run with sudo (some JetPack versions restrict
tegrastats counters for non-root users).
"""

from __future__ import annotations

import argparse
import json
import re
import signal
import subprocess
import tempfile
import time
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]

PATTERNS = {
    "gpu_power_mw": [
        ("VDD_GPU_SOC", re.compile(r"VDD_GPU_SOC (\d+)mW")),
        ("POM_5V_GPU", re.compile(r"POM_5V_GPU (\d+)mW")),
        # some Orin NX boards expose no dedicated GPU rail; the combined
        # CPU+GPU+CV rail is then the closest available (upper bound for GPU)
        ("VDD_CPU_GPU_CV", re.compile(r"VDD_CPU_GPU_CV (\d+)mW")),
    ],
    "total_power_mw": [
        ("VDD_IN", re.compile(r"VDD_IN (\d+)mW")),
        ("POM_5V_IN", re.compile(r"POM_5V_IN (\d+)mW")),
    ],
    "gpu_temp_c": [("GPU", re.compile(r"GPU@([\d.]+)C"))],
}
RAM_RE = re.compile(r"RAM (\d+)/(\d+)MB")


def parse_tegrastats(path: Path) -> tuple[dict, dict]:
    samples = {"gpu_power_mw": [], "total_power_mw": [], "gpu_temp_c": [], "ram_used_mb": [], "ram_total_mb": []}
    sources = {key: None for key in PATTERNS}
    for line in path.read_text(errors="replace").splitlines():
        for key, patterns in PATTERNS.items():
            for name, pattern in patterns:
                m = pattern.search(line)
                if m:
                    samples[key].append(float(m.group(1)))
                    sources[key] = name
                    break
        m = RAM_RE.search(line)
        if m:
            samples["ram_used_mb"].append(int(m.group(1)))
            samples["ram_total_mb"].append(int(m.group(2)))
    return samples, sources


def stats(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "max": None, "min": None, "n": 0}
    return {
        "mean": round(sum(values) / len(values), 1),
        "max": round(max(values), 1),
        "min": round(min(values), 1),
        "n": len(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--interval-ms", type=int, default=200)
    parser.add_argument("--settle-s", type=float, default=2.0, help="baseline sampling before the command starts")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("command", nargs=argparse.REMAINDER, help="command after --")
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        raise SystemExit("no command given; put it after --")

    tmp = tempfile.NamedTemporaryFile(prefix="tegrastats_", suffix=".log", delete=False)
    tmp.close()
    log_path = Path(tmp.name)
    # non-root tegrastats omits the power rails on this JetPack, so the sampler
    # must run as root; root tegrastats ignores --logfile and prints to stdout
    # instead, so capture its stdout into the log file ourselves
    log_fh = open(log_path, "w")
    proc = subprocess.Popen(
        ["sudo", "-n", "tegrastats", "--interval", str(args.interval_ms)],
        stdout=log_fh,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(max(args.settle_s, 0.1))

    t0 = time.perf_counter()
    run = subprocess.run(args.command)
    duration = time.perf_counter() - t0

    time.sleep(max(args.settle_s, 0.1))
    # tegrastats runs as a root grandchild (sudo -> tegrastats) that an
    # unprivileged parent cannot signal: target the wrapper's children
    # directly, with a cmdline fallback in case sudo exec'd tegrastats
    def stop_sampler(sig: str) -> None:
        for pattern in (["-P", str(proc.pid)], ["-f", "tegrastats --interval"]):
            subprocess.run(
                ["sudo", "-n", "pkill", sig] + pattern,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )

    stop_sampler("-INT")
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        stop_sampler("-KILL")
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            report_warning = "tegrastats sampler did not exit; extra trailing samples possible"
        else:
            report_warning = None
    else:
        report_warning = None
    log_fh.close()

    samples, sources = parse_tegrastats(log_path)
    temps = samples["gpu_temp_c"]
    stable_temp = round(sum(temps[-10:]) / max(len(temps[-10:]), 1), 1) if temps else None
    gpu_note = "Orin: VDD_GPU_SOC covers the GPU+SoC rail; Xavier: POM_5V_GPU."
    if sources["gpu_power_mw"] == "VDD_CPU_GPU_CV":
        gpu_note += (
            " THIS BOARD HAS NO DEDICATED GPU RAIL: gpu_power_mw is the combined"
            " CPU+GPU+CV rail, an upper bound for GPU power - state this in the thesis."
        )
    report = {
        "label": args.label,
        "command": args.command,
        "returncode": run.returncode,
        "duration_s": round(duration, 2),
        "tegrastats_interval_ms": args.interval_ms,
        "gpu_power_source": sources["gpu_power_mw"],
        "total_power_source": sources["total_power_mw"],
        "gpu_power_mw": stats(samples["gpu_power_mw"]),
        "total_power_mw": stats(samples["total_power_mw"]),
        "gpu_temp_c": {**stats(temps), "stable_last10_mean": stable_temp},
        "ram_used_mb_max": max(samples["ram_used_mb"]) if samples["ram_used_mb"] else None,
        "ram_total_mb": samples["ram_total_mb"][-1] if samples["ram_total_mb"] else None,
        "field_note": gpu_note + " Peak memory on Jetson is unified RAM, not a separate GPU pool.",
        "raw_log": str(log_path),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    if report_warning:
        report["warning"] = report_warning
    if run.returncode != 0:
        report["warning"] = "wrapped command failed; power/temp samples are still recorded"
    out_path = args.output or (PACKAGE_ROOT / "results" / f"power_{args.label}_{time.strftime('%Y%m%d_%H%M%S')}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("label", "gpu_power_mw", "total_power_mw", "gpu_temp_c", "ram_used_mb_max")}, indent=2))
    print(f"[record] {out_path}")
    raise SystemExit(run.returncode)


if __name__ == "__main__":
    main()
