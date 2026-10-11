"""Approval after quality + measured serving gates; rollback to approved versions."""
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

from food_delivery.train import write_json
from food_delivery.evidence import identity


def approve(out="artifacts"):
    out = Path(out).resolve()
    quality = json.loads((out/"gate.json").read_text())
    serving = json.loads((out/"benchmark.json").read_text())
    size = (out/"candidate.joblib").stat().st_size
    if not quality["passed"] or not serving["passed"] or size > 100*1024**2:
        raise ValueError("Quality, measured serving, or 100MB size gate failed")
    if serving.get("identity") != identity(out):
        raise ValueError("HTTP evidence is stale or belongs to a different candidate")
    mlflow.set_tracking_uri(f"sqlite:///{(out/'mlflow.db').as_posix()}")
    client = MlflowClient()
    name, version = quality["registry_name"], quality["version"]
    if client.get_model_version(name, str(version)).run_id != quality["run_id"]:
        raise ValueError("Registry version belongs to another training run")
    bundle = joblib.load(out/"candidate.joblib")
    bundle.metadata.update({"version": version, "registry_name": name, "approval": "approved_local_demo"})
    staged = out/"model.next.joblib"
    joblib.dump(bundle, staged)
    previous = (out/"model.joblib").read_bytes() if (out/"model.joblib").exists() else None
    staged.replace(out/"model.joblib")
    try:
        client.set_registered_model_alias(name,"champion",version)
    except Exception:
        if previous is not None:
            staged.write_bytes(previous)
            staged.replace(out/"model.joblib")
        else:
            (out/"model.joblib").unlink()
        raise
    client.set_model_version_tag(name, version, "approval_status", "approved_local_demo")
    result = {"registry_name":name,"version":version,"artifact_bytes":size,"status":"approved_local_demo",
              "scope":"local quality and HTTP gates; not evidence of container or cloud deployment"}
    write_json(out/"approval.json",result)
    return result


def rollback(name, version, out="artifacts"):
    out = Path(out).resolve()
    mlflow.set_tracking_uri(f"sqlite:///{(out/'mlflow.db').as_posix()}")
    client = MlflowClient()
    record = client.get_model_version(name, str(version))
    if record.tags.get("approval_status") != "approved_local_demo":
        raise ValueError("Rollback target must be approved")
    bundle = mlflow.sklearn.load_model(f"models:/{name}/{version}")
    bundle.metadata.update({"version":str(version),"registry_name":name,"approval":"approved_local_demo"})
    staged = out/"model.next.joblib"
    joblib.dump(bundle, staged)
    previous = (out/"model.joblib").read_bytes() if (out/"model.joblib").exists() else None
    staged.replace(out/"model.joblib")
    try:
        client.set_registered_model_alias(name,"champion",str(version))
    except Exception:
        if previous is not None:
            staged.write_bytes(previous)
            staged.replace(out/"model.joblib")
        else:
            (out/"model.joblib").unlink()
        raise
    return {"champion":str(version),"restart_required":True}
