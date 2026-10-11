"""Read local credentials without displaying them; check existing license access."""
import json
from pathlib import Path
from urllib.parse import quote

import requests

out = Path("artifacts-tabpfn-gpu")
out.mkdir(exist_ok=True)
token = Path(".cache-tabpfn/token.local").read_text(encoding="utf-8-sig").strip()
report = {"token_present": bool(token), "repo": "Prior-Labs/tabpfn_3_5"}
try:
    verified = requests.get("https://api.priorlabs.ai/protected/", headers={"Authorization": f"Bearer {token}"}, timeout=20)
    report["token_http_status"] = verified.status_code
    report["token_valid"] = verified.status_code == 200
    if report["token_valid"]:
        meta = requests.get("https://huggingface.co/api/models/Prior-Labs/tabpfn_3_5", timeout=20)
        report["model_metadata_http_status"] = meta.status_code
        meta.raise_for_status()
        license_name = meta.json().get("cardData", {}).get("license_name")
        if not license_name:
            raise ValueError("Model metadata did not provide license_name")
        check = requests.get("https://api.priorlabs.ai/account/license/?version=" + quote(license_name, safe=""),
                             headers={"Authorization": f"Bearer {token}"}, timeout=20)
        report["license_accepted"] = check.status_code == 200 and check.json().get("accepted", False)
    report["status"] = "PASSED" if report.get("token_valid") and report.get("license_accepted") else "BLOCKED"
except requests.RequestException:
    report.update(status="NETWORK_FAILED", reason="Authentication/model metadata endpoint unreachable")
(out / "access_check.json").write_text(json.dumps(report, indent=2), encoding="utf8")
print(json.dumps(report, indent=2))
if report["status"] != "PASSED":
    raise SystemExit(1)
