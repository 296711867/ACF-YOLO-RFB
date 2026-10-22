"""Collect the Jetson environment ledger (deployment-spec §16.3).

Gathers board/JetPack/CUDA/cuDNN/TensorRT versions, power mode and clocks,
memory and disk. Commands that require root are detected and reported as
"needs sudo" hints instead of failing. Writes results/jetson_env_audit.json
and prints a Markdown table that can be pasted straight into the thesis
environment ledger.
"""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import time
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if out.returncode != 0:
            return (out.stderr or "").strip() or None
        return out.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError):
        return None


def python_module_version(name: str) -> str | None:
    try:
        module = __import__(name)
        return str(getattr(module, "__version__", "present"))
    except Exception as exc:  # noqa: BLE001 - report any import failure verbatim
        return f"IMPORT FAILED: {exc}"


def sample_tegrastats_once() -> str | None:
    tmp = Path(time.strftime("/tmp/tegrastats_audit_%s.log"))
    try:
        proc = subprocess.Popen(
            ["tegrastats", "--interval", "500", "--logfile", str(tmp)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(1.5)
        proc.terminate()
        proc.wait(timeout=5)
        return tmp.read_text(errors="replace").splitlines()[0] if tmp.is_file() else None
    except Exception:  # noqa: BLE001
        return None


def main() -> None:
    ledger = {
        "collected_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "board": {
            "device_tree_model": (Path("/proc/device-tree/model").read_text(errors="replace").strip("\x00 ") or None)
            if Path("/proc/device-tree/model").exists()
            else None,
            "nv_tegra_release": (Path("/etc/nv_tegra_release").read_text(errors="replace").strip() or None)
            if Path("/etc/nv_tegra_release").exists()
            else None,
        },
        "os": {
            "uname": run(["uname", "-a"]),
            "python": platform.python_version(),
            "gcc": (run(["gcc", "--version"]) or "").splitlines()[0] if run(["gcc", "--version"]) else None,
        },
        "jetpack": {
            "nvidia_jetpack": run(["dpkg-query", "-W", "-f=${Version}", "nvidia-jetpack"]),
            "cuda_dpkg": run(["dpkg-query", "-W", "-f=${Version}", "cuda-toolkit-12-6"])
            or run(["dpkg-query", "-W", "-f=${Version}", "cuda-toolkit-12-4"])
            or run(["dpkg-query", "-W", "-f=${Version}", "nvidia-l4t-cuda"]),
            "cudnn_dpkg": run(["dpkg-query", "-W", "-f=${Version}", "nvidia-l4t-cudnn9"])
            or run(["dpkg-query", "-W", "-f=${Version}", "nvidia-l4t-cudnn8"]),
            "tensorrt_dpkg": run(["dpkg-query", "-W", "-f=${Version}", "nvidia-l4t-tensorrt"]),
            "opencv_dpkg": run(["dpkg-query", "-W", "-f=${Version}", "libopencv-python"]),
        },
        "python_modules": {
            "tensorrt": python_module_version("tensorrt"),
            "pycuda": python_module_version("pycuda"),
            "numpy": python_module_version("numpy"),
            "cv2": python_module_version("cv2"),
        },
        "power_and_clocks": {
            "nvpmodel": run(["nvpmodel", "-q"]),
            "nvpmodel_note": "if empty, run: sudo nvpmodel -q",
            "jetson_clocks_show": run(["sudo", "-n", "jetson_clocks", "--show"]),
            "jetson_clocks_note": "if empty, run: sudo jetson_clocks --show (or sudo jetson_clocks to lock max clocks)",
        },
        "memory_disk": {"free": run(["free", "-h"]), "df_root": run(["df", "-h", "/"])},
        "tegrastats_first_sample": sample_tegrastats_once(),
        "trtexec_path": shutil.which("trtexec") or ("/usr/src/tensorrt/bin/trtexec" if Path("/usr/src/tensorrt/bin/trtexec").exists() else None),
    }

    out_path = PACKAGE_ROOT / "results" / "jetson_env_audit.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")

    def kv(title, value):
        value = (value or "n/a").replace("\n", "; ").replace("|", "\\|")
        print(f"| {title} | {value} |")

    print("\n## Jetson 环境台账（论文§16.3）\n")
    print("| 项目 | 值 |")
    print("|---|---|")
    kv("板卡型号", ledger["board"]["device_tree_model"])
    kv("L4T release", ledger["board"]["nv_tegra_release"])
    kv("JetPack", ledger["jetpack"]["nvidia_jetpack"])
    kv("CUDA", ledger["jetpack"]["cuda_dpkg"])
    kv("cuDNN", ledger["jetpack"]["cudnn_dpkg"])
    kv("TensorRT (apt)", ledger["jetpack"]["tensorrt_dpkg"])
    kv("TensorRT (python)", ledger["python_modules"]["tensorrt"])
    kv("pycuda", ledger["python_modules"]["pycuda"])
    kv("numpy / cv2", f'{ledger["python_modules"]["numpy"]} / {ledger["python_modules"]["cv2"]}')
    kv("电源模式", ledger["power_and_clocks"]["nvpmodel"])
    kv("tegrastats 首样本", ledger["tegrastats_first_sample"])
    kv("trtexec", ledger["trtexec_path"])
    print(f"\n[record] {out_path}")


if __name__ == "__main__":
    main()
