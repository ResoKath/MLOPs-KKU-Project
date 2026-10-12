"""Bind promotion evidence to one immutable candidate and recompute metrics."""
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from food_delivery.preprocessing import TARGET, prepare_data
from food_delivery.train import batched_predict, gate, metrics


def identity(out):
    out = Path(out)
    decision = json.loads((out / "gate.json").read_text())
    bundle = joblib.load(out / "candidate.joblib")
    if bundle.metadata.get("run_id") != decision.get("run_id"):
        raise ValueError("Candidate run differs from quality evidence")
    digest = hashlib.sha256((out / "candidate.joblib").read_bytes()).hexdigest()
    if digest != decision.get("candidate_sha256"):
        raise ValueError("Candidate hash differs from quality evidence")
    return {key: decision[key] for key in ("run_id", "registry_name", "version", "candidate_sha256", "data_sha256")}


def verify_quality(data, out, degrade=False):
    out = Path(out)
    bound = identity(out)
    if hashlib.sha256(Path(data).read_bytes()).hexdigest() != bound["data_sha256"]:
        raise ValueError("Dataset differs from quality evidence")
    (_, val, test), _ = prepare_data(data)
    bundle = joblib.load(out / "candidate.joblib")
    measured = {}
    for name, frame in [("validation", val), ("test", test)]:
        predictions = batched_predict(bundle, frame)
        if degrade:
            predictions = predictions + 20  # Actual degraded predictions, not a stored boolean.
        measured[name] = metrics(frame[TARGET], predictions)
    selection = json.loads((out / "selection.json").read_text())
    decision = json.loads((out / "gate.json").read_text())
    passed = gate(measured["validation"], selection["validation"]["dummy"]["mae"],
                  measured["test"], decision["max_mae"])
    if not passed:
        raise ValueError(f"Recomputed model quality gate failed: {measured}")
    if not np.isclose(measured["validation"]["mae"], selection["validation"][selection["selected"]]["mae"]):
        raise ValueError("Recomputed validation metrics differ from logged evidence")
    return {"identity": bound, "recomputed": measured, "passed": passed}
