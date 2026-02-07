from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys


def run(cmd: list[str]) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def resolve_nvidia_smi() -> str | None:
    candidates = [
        shutil.which("nvidia-smi"),
        "/usr/lib/wsl/lib/nvidia-smi",
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None


def check_torch() -> dict:
    result = {
        "import_ok": False,
        "version": None,
        "cuda_available": None,
        "cuda_device_count": None,
        "devices": [],
        "error": None,
    }
    try:
        import torch

        result["import_ok"] = True
        result["version"] = torch.__version__
        result["cuda_available"] = torch.cuda.is_available()
        result["cuda_device_count"] = torch.cuda.device_count()
        result["devices"] = [
            {
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "total_memory_gb": round(torch.cuda.get_device_properties(i).total_memory / (1024**3), 2),
            }
            for i in range(torch.cuda.device_count())
        ]
    except Exception as exc:  # pragma: no cover
        result["error"] = str(exc)
    return result


def main() -> int:
    report = {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "python": sys.version.split()[0],
            "is_wsl": "microsoft" in platform.release().lower() or os.path.exists("/dev/dxg"),
            "dxg_present": os.path.exists("/dev/dxg"),
        },
        "nvidia_smi": {
            "path": None,
            "ok": False,
            "summary": None,
            "error": None,
        },
        "torch": check_torch(),
    }

    smi = resolve_nvidia_smi()
    report["nvidia_smi"]["path"] = smi
    if smi:
        rc, out, err = run(
            [
                smi,
                "--query-gpu=index,name,memory.total,driver_version",
                "--format=csv,noheader",
            ]
        )
        report["nvidia_smi"]["ok"] = rc == 0
        report["nvidia_smi"]["summary"] = out or None
        report["nvidia_smi"]["error"] = err or None
    else:
        report["nvidia_smi"]["error"] = "nvidia-smi not found in PATH or /usr/lib/wsl/lib"

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
