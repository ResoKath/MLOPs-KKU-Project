"""Load native fitted state with an explicitly relocated checkpoint and raw JSON."""
# Workspace caches and project import path precede third-party imports.
# ruff: noqa: E402
import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("SKB_DATA_DIRECTORY", str(ROOT / ".cache-tabpfn/skrub"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache-tabpfn/matplotlib"))
os.environ.setdefault("TABPFN_NO_BROWSER", "1")

import pandas as pd

from food_delivery.tabpfn_model import RawTabPFNModel


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", default="artifacts-tabpfn-gpu/example_request_records.json")
    parser.add_argument("--checkpoint", default=".cache-tabpfn/models/tabpfn-v3.5-20260909.safetensors")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    os.environ["TABPFN_SERVING_DEVICE"] = args.device
    out = ROOT / "artifacts-tabpfn-gpu"
    context = SimpleNamespace(model_config={"batch_size": 16}, artifacts={
        "fitted_state": str(out / "model.tabpfn_fit"),
        "preprocessing": str(out / "preprocessing_config.json"),
        "checkpoint": str(Path(args.checkpoint).resolve()),
    })
    model = RawTabPFNModel()
    model.load_context(context)
    records = pd.DataFrame(json.loads(Path(args.records).read_text()))
    prediction = model.predict(context, records)
    result = {"status": "PASSED", "rows": len(records), "predictions_minutes": prediction.tolist(),
              "scope": "native fitted state + relocated checkpoint + shared raw preprocessing; no refitting"}
    (out / "saved_model_prediction_check.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps(result, indent=2))
