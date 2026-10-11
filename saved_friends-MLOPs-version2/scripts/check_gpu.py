"""Verify CUDA availability with an actual tensor operation, then save evidence."""
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import torch

output = Path("artifacts-tabpfn-gpu")
output.mkdir(exist_ok=True)
report = {
    "checked_at": datetime.now(timezone.utc).isoformat(),
    "python": platform.python_version(),
    "torch": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
}
if not report["cuda_available"]:
    report["status"] = "FAILED"
else:
    device = torch.cuda.get_device_properties(0)
    report.update(gpu=device.name, total_vram_bytes=device.total_memory)
    free, total = torch.cuda.mem_get_info()
    report.update(free_vram_bytes=free)
    torch.manual_seed(42)
    a = torch.randn(256, 256, device="cuda")
    b = torch.randn(256, 256, device="cuda")
    result = a @ b
    torch.cuda.synchronize()
    torch.testing.assert_close(result.cpu(), a.cpu() @ b.cpu(), rtol=1e-3, atol=1e-3)
    report.update(status="PASSED", tensor_device=str(result.device), operation="256x256 matrix multiplication; checked against CPU")
(output / "gpu_check.json").write_text(json.dumps(report, indent=2), encoding="utf8")
print(json.dumps(report, indent=2))
if report["status"] != "PASSED":
    raise SystemExit(1)
