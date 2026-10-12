"""Repeatable Prefect DAG: raw -> validation -> training -> HTTP -> approval -> monitoring."""
import argparse
import json
import os
import time
from pathlib import Path

from prefect import flow, task

from food_delivery.serving import serving_gate as measure_serving
from food_delivery.monitor import from_events
from food_delivery.preprocessing import prepare_data
from food_delivery.registry import approve
from food_delivery.train import run, write_json


@task(name="validate-and-temporal-split", retries=0)
def validate(data):
    _, quality = prepare_data(data)
    return quality


@task(name="fit-preprocess-train-compare-evaluate-register", retries=0)
def train(data, output, tabpfn):
    return run(data,out=output,with_tabpfn=tabpfn)


serving_gate = task(name="serve-and-measure-http", retries=0)(measure_serving)


@task(name="approve-local-champion", retries=0)
def approval(output):
    return approve(output)


@task(name="monitor-recent-events", retries=0)
def monitoring(output):
    output = Path(output)
    selection = json.loads((output/"selection.json").read_text())
    return from_events(output/"reference_features.csv",output/"events.db",output/"monitor_live.json",
                       selection["validation"][selection["selected"]]["mae"])


@flow(name="food-delivery-end-to-end", log_prints=True)
def delivery_flow(data="input/Zomato Dataset.csv",output="artifacts",tabpfn=False):
    quality = validate(data)
    candidate = train(data,output,tabpfn)
    measured = serving_gate(output)
    approved = approval(output)
    monitor = monitoring(output)
    evidence = {"data":quality,"candidate":candidate,"http":measured,"approval":approved,"monitor":monitor,
                "dag":"validate -> fit/compare/test/registry -> HTTP measure -> approve -> monitor"}
    write_json(Path(output)/"pipeline_result.json",evidence)
    return evidence


def local_dag(data="input/Zomato Dataset.csv",output="artifacts",tabpfn=False):
    """Small explicit DAG executor; fallback when Prefect server cannot start."""
    nodes = {
        "validate": ([], lambda: validate.fn(data)),
        "train_evaluate_register": (["validate"], lambda: train.fn(data,output,tabpfn)),
        "http_gate": (["train_evaluate_register"], lambda: serving_gate.fn(output)),
        "approve": (["http_gate"], lambda: approval.fn(output)),
        "monitor": (["approve"], lambda: monitoring.fn(output)),
    }
    results, states = {}, {}
    evidence = {"engine":"local explicit DAG executor", "edges":{k:v[0] for k,v in nodes.items()},"states":states}
    while len(results) < len(nodes):
        ready = [key for key,(deps,_) in nodes.items() if key not in results and all(dep in results for dep in deps)]
        if not ready:
            raise ValueError("DAG has a cycle or missing dependency")
        for key in ready:
            start = time.perf_counter()
            states[key] = {"status":"RUNNING"}
            write_json(Path(output)/"dag_state.json",evidence)
            try:
                results[key] = nodes[key][1]()
            except Exception as error:
                states[key] = {"status":"FAILED","error":str(error)}
                write_json(Path(output)/"dag_state.json",evidence)
                raise
            states[key] = {"status":"PASSED","seconds":time.perf_counter()-start}
            write_json(Path(output)/"dag_state.json",evidence)
    evidence["results"] = results
    write_json(Path(output)/"pipeline_result.json",evidence)
    return evidence


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data",default="input/Zomato Dataset.csv")
    p.add_argument("--out",default="artifacts")
    p.add_argument("--tabpfn",action="store_true")
    p.add_argument("--serve",action="store_true",help="After passing the DAG, keep the API serving on port 8000")
    p.add_argument("--engine",choices=["local","prefect"],default="local")
    a = p.parse_args()
    if a.engine == "prefect":
        delivery_flow(a.data,a.out,a.tabpfn)
    else:
        local_dag(a.data,a.out,a.tabpfn)
    if a.serve:
        os.environ["MODEL_PATH"] = str(Path(a.out).resolve()/"model.joblib")
        os.environ["EVENT_DB"] = str(Path(a.out).resolve()/"events.db")
        import uvicorn
        from food_delivery.api import create_app
        uvicorn.run(create_app(),host="0.0.0.0",port=8000)
