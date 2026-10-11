"""Isolated synthetic registry cycle; host restarts the real container between stages."""
import argparse
import os
import shutil
from pathlib import Path

import pandas as pd

from food_delivery.pipeline import local_dag
from food_delivery.preprocessing import TARGET
from food_delivery.registry import rollback
from food_delivery.train import write_json


def stage(action):
    live = Path("artifacts")
    demo = live / "synthetic-demo"
    demo.mkdir(exist_ok=True)
    os.environ["MODEL_REGISTRY_SUFFIX"] = "-synthetic-demo"
    if action == "baseline":
        shutil.copy2(live/"model.joblib",demo/"original-live.joblib")
        result = local_dag(output=str(demo))
        shutil.copy2(demo/"model.joblib",live/"model.joblib")
        write_json(demo/"cycle-baseline.json",result)
    elif action == "retrain":
        if not (live/"retrain_request.json").exists():
            raise ValueError("Monitoring must request retraining before this demonstration")
        # Controlled +12-minute concept shift. This is not newly collected business data.
        raw = pd.read_csv("input/Zomato Dataset.csv")
        values = pd.to_numeric(raw[TARGET].astype(str).str.replace("(min)","",regex=False).str.strip())
        raw[TARGET] = values + 12
        raw.to_csv(demo/"synthetic-shift.csv",index=False)
        result = local_dag(str(demo/"synthetic-shift.csv"),str(demo))
        shutil.copy2(demo/"model.joblib",live/"model.joblib")
        write_json(demo/"cycle-retrain.json",result)
    elif action == "rollback":
        import json
        name = json.loads((demo/"approval.json").read_text())["registry_name"]
        result = rollback(name,"1",str(demo))
        shutil.copy2(demo/"model.joblib",live/"model.joblib")
        write_json(demo/"cycle-rollback.json",result)
    elif action == "restore":
        shutil.copy2(demo/"original-live.joblib",live/"model.joblib")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action",choices=["baseline","retrain","rollback","restore"])
    stage(parser.parse_args().action)
