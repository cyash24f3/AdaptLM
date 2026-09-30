"""No CUDA packages or model download required for fixture checks."""

import json
import platform
import shutil
import subprocess

import psutil

report = {
    "platform": platform.platform(),
    "python": platform.python_version(),
    "memory_bytes": psutil.virtual_memory().total,
    "disk_free_bytes": shutil.disk_usage(".").free,
}
try:
    import torch

    report.update(
        torch=torch.__version__,
        mps=torch.backends.mps.is_available(),
        cuda=torch.cuda.is_available(),
    )
    if report["mps"]:
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            x = torch.ones(4, 4, device="mps", dtype=dtype, requires_grad=True)
            (x @ x).sum().backward()
            report[str(dtype)] = bool(torch.isfinite(x.grad).all())
        report["gpu"] = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        ).stdout.strip()
except ImportError:
    report["torch"] = "optional model profile not installed"
print(json.dumps(report, indent=2))
