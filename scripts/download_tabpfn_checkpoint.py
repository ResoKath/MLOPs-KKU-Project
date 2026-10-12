"""Download the official default checkpoint after an existing license access check."""
import hashlib
import json
from pathlib import Path

import requests

access = json.loads(Path("artifacts-tabpfn-gpu/access_check.json").read_text())
if access["status"] != "PASSED":
    raise SystemExit("Existing license access check must pass first")
directory = Path(".cache-tabpfn/models")
directory.mkdir(parents=True, exist_ok=True)
name = "tabpfn-v3.5-20260909.safetensors"
url = "https://huggingface.co/Prior-Labs/tabpfn_3_5/resolve/main/" + name
target = directory / name
if not target.exists():
    partial = target.with_suffix(".partial")
    with requests.get(url, stream=True, timeout=(30, 180)) as response:
        if response.status_code != 200:
            raise SystemExit(f"Checkpoint HTTP status: {response.status_code}; no credential sent to Hugging Face")
        total = int(response.headers.get("content-length", 0))
        received, next_update = 0, 64 * 1024**2
        with partial.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024**2):
                handle.write(chunk)
                received += len(chunk)
                if received >= next_update:
                    print(f"Checkpoint downloaded: {received / 1024**2:.0f} MiB" + (f" / {total / 1024**2:.0f} MiB" if total else ""), flush=True)
                    next_update += 64 * 1024**2
        if total and received != total:
            raise SystemExit("Incomplete checkpoint transfer")
        partial.replace(target)
digest = hashlib.sha256()
with target.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024**2), b""):
        digest.update(chunk)
report = {"status": "PASSED", "repo": "Prior-Labs/tabpfn_3_5", "checkpoint": name,
          "bytes": target.stat().st_size, "sha256": digest.hexdigest(), "source": url}
Path("artifacts-tabpfn-gpu/checkpoint.json").write_text(json.dumps(report, indent=2), encoding="utf8")
print(json.dumps(report, indent=2))
