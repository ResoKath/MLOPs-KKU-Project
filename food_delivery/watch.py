"""Container monitoring loop. Emits a retraining request, never auto-promotes."""
import json
import time
from pathlib import Path

from food_delivery.monitor import from_events
from food_delivery.train import write_json


def tick(out="artifacts"):
    out = Path(out)
    selection = json.loads((out/"selection.json").read_text())
    baseline_mae = selection["validation"][selection["selected"]]["mae"]
    state_path = out/"monitor_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"degraded_windows":0,"consumed_labels":[]}
    approval = json.loads((out/"approval.json").read_text())
    active_version = f"{approval['registry_name']}:{approval['version']}"
    if state.get("model_version") != active_version:
        state = {"degraded_windows":0,"consumed_labels":[],"model_version":active_version}
    result = from_events(out/"reference_features.csv",out/"events.db",out/"monitor_live.json",baseline_mae,
                         model_version=active_version, consumed_labels=state.get("consumed_labels",[]))
    if result["label_rows"] >= 100:
        state["degraded_windows"] = state["degraded_windows"]+1 if result.get("retrain_requested") else 0
        state["consumed_labels"] = list(set(state.get("consumed_labels",[])) | set(result["label_request_ids"]))
        state["last_quality"] = {key: result[key] for key in ["quality_status", "mae", "quality_threshold", "concept_drift_signal", "retrain_requested"] if key in result}
    elif state.get("last_quality"):
        result.update(state["last_quality"])
        result["quality_scope"] = "last completed non-overlapping label window; waiting for 100 new labels"
        write_json(out/"monitor_live.json", result)
    write_json(state_path,state)
    if state["degraded_windows"] >= 2:
        write_json(out/"retrain_request.json",{"reason":"delayed-label MAE degraded", "status":"review fresh labeled data and run pipeline",
                   "policy":"minimum 100 new labels per non-overlapping window; label verification requires human review; preserve new chronological test"})
    return result


if __name__ == "__main__":
    while True:
        try:
            tick()
        except (FileNotFoundError, OSError) as error:
            print(f"Monitoring waiting for artifacts: {error}",flush=True)
        time.sleep(60)
