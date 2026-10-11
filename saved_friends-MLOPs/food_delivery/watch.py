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
    result = from_events(out/"reference_features.csv",out/"events.db",out/"monitor_live.json",baseline_mae)
    state_path = out/"monitor_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"degraded_windows":0,"previous_window":None}
    if result["label_window_id"] != state["previous_window"]:
        state["degraded_windows"] = state["degraded_windows"]+1 if result.get("retrain_requested") else 0
        state["previous_window"] = result["label_window_id"]
    write_json(state_path,state)
    if state["degraded_windows"] >= 2:
        write_json(out/"retrain_request.json",{"reason":"delayed-label MAE degraded", "status":"review fresh labeled data and run pipeline",
                   "policy":"minimum 100 verified labels; two consecutive distinct degraded windows; preserve new chronological test"})
    return result


if __name__ == "__main__":
    while True:
        try:
            tick()
        except (FileNotFoundError, OSError) as error:
            print(f"Monitoring waiting for artifacts: {error}",flush=True)
        time.sleep(60)
