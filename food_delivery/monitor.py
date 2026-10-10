"""Data drift and delayed-label quality degradation; no causal drift claims."""
import argparse
import hashlib
import json
import sqlite3

import numpy as np
import pandas as pd

from food_delivery.preprocessing import CATEGORIES, FEATURES
from food_delivery.train import write_json


def psi(reference, current, categorical=False):
    if categorical:
        support = sorted(set(reference.astype(str)) | set(current.astype(str)))
        a = reference.astype(str).value_counts(normalize=True).reindex(support, fill_value=0).values
        b = current.astype(str).value_counts(normalize=True).reindex(support, fill_value=0).values
    else:
        edges = np.unique(np.quantile(reference.astype(float), np.linspace(0,1,11)))
        if len(edges) < 2:
            edges = np.array([-np.inf, float(reference.iloc[0])+1e-6, np.inf])
        else:
            edges[0], edges[-1] = -np.inf, np.inf
        a = np.histogram(reference.astype(float), bins=edges)[0] / len(reference)
        b = np.histogram(current.astype(float), bins=edges)[0] / len(current)
    a, b = np.maximum(a, 1e-6), np.maximum(b, 1e-6)
    return float(np.sum((b-a)*np.log(b/a)))


def evaluate(reference, current, predictions=None, actual=None, reference_mae=None):
    if len(current) < 100:
        return {"status": "insufficient_data", "rows": len(current), "minimum_rows": 100}
    scores = {col: psi(reference[col],current[col],col in CATEGORIES) for col in FEATURES}
    drifted = [col for col, value in scores.items() if value >= .2]
    result = {"status": "evaluated", "rows": len(current), "psi": scores,
              "data_drift": bool(drifted), "drifted_features": drifted,
              "quality_status": "labels_unavailable", "retrain_requested": False}
    if actual is not None and len(actual) >= 100:
        residual = np.asarray(actual) - np.asarray(predictions)
        mae = float(np.mean(np.abs(residual)))
        threshold = max(5.0, reference_mae * 1.2) if reference_mae else 5.0
        degraded = mae > threshold
        result.update({"mae": mae, "mean_residual": float(np.mean(residual)), "quality_threshold": threshold,
                       "quality_status": "degraded" if degraded else "healthy",
                       "concept_drift_signal": bool(degraded),
                       "interpretation": "label-based degradation signal; investigate changes in P(y|X), label quality and service conditions",
                       "retrain_requested": bool(degraded)})
    result["alert"] = bool(drifted) or result["retrain_requested"]
    return result


def from_events(reference_path, db_path, out_path, reference_mae, limit=1000):
    reference = pd.read_csv(reference_path)
    with sqlite3.connect(db_path) as db:
        rows = db.execute("SELECT features,predictions,actual,request_id FROM events ORDER BY created DESC LIMIT ?", (limit,)).fetchall()
    current, predictions, actual = [], [], []
    for feature_json, pred_json, actual_json, request_id in rows:
        current.extend(json.loads(feature_json))
        if actual_json:
            predictions.extend(json.loads(pred_json))
            actual.extend(json.loads(actual_json))
    result = evaluate(reference, pd.DataFrame(current), predictions, actual or None, reference_mae)
    result["window"] = f"latest {limit} requests; minimum 100 feature/label rows"
    result["label_window_id"] = hashlib.sha256(json.dumps([(r[3],r[2]) for r in rows if r[2]]).encode()).hexdigest()
    result["label_rows"] = len(actual)
    write_json(out_path, result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--reference", default="artifacts/reference_features.csv")
    p.add_argument("--events", default="artifacts/events.db")
    p.add_argument("--out", default="artifacts/monitor_live.json")
    p.add_argument("--reference-mae", type=float, default=5.0)
    a = p.parse_args()
    print(json.dumps(from_events(a.reference,a.events,a.out,a.reference_mae), indent=2))
