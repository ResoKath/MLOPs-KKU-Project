"""Run real host-to-container acceptance tests; synthetic labels are explicit."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from food_delivery.benchmark import benchmark  # noqa: E402
from food_delivery.train import write_json  # noqa: E402

OUT = ROOT / "evidence"
BASE = "http://127.0.0.1:8000"
PROM = "http://127.0.0.1:9090"


def docker(*args):
    subprocess.run(["docker","compose",*args],cwd=ROOT,check=True)


def ready():
    for _ in range(90):
        try:
            response = httpx.get(BASE+"/health/ready",timeout=2)
            if response.status_code == 200:
                return response.json()
        except httpx.HTTPError:
            pass
        time.sleep(1)
    raise RuntimeError("Container API readiness timed out")


def records():
    return json.loads((ROOT/"artifacts-container/example_request_records.json").read_text())


def prediction():
    response = httpx.post(BASE+"/predict",json={"records":records()},timeout=30)
    response.raise_for_status()
    return response.json()


def baseline():
    health = ready()
    result = benchmark(BASE,str(ROOT/"artifacts-container/example_request_records.json"),str(OUT/"container_benchmark.json"))
    result["scope"] = "Windows host to Linux Docker published port; 100 requests, concurrency 4, one record/request"
    write_json(OUT/"container_benchmark.json",result)
    assert result["passed"], result
    invalid = [httpx.post(BASE+"/predict",json=payload).status_code for payload in [{"records":[{}]},{"records":[]},{"records":[{}]*101}]]
    assert invalid == [422]*3
    targets = httpx.get(PROM+"/api/v1/targets").json()
    assert any(t["health"] == "up" for t in targets["data"]["activeTargets"])
    write_json(OUT/"container_acceptance.json",{"health":health,"bad_payload_statuses":invalid,"prometheus_targets":targets,"scope":"actual published port; no second physical machine or internet deployment"})
    (OUT/"container_metrics.txt").write_text(httpx.get(BASE+"/metrics").text)
    print(json.dumps(result,indent=2),flush=True)


def feedback():
    record = records()[0]
    record.update({"Restaurant_latitude":18.52,"Restaurant_longitude":73.85,
                   "Delivery_location_latitude":18.78,"Delivery_location_longitude":74.10})
    windows = []
    for _ in range(2):
        response = httpx.post(BASE+"/predict",json={"records":[record]*100},timeout=30)
        response.raise_for_status()
        result = response.json()
        labels = [value+12 for value in result["predictions_minutes"]]
        posted = httpx.post(BASE+"/feedback",json={"request_id":result["request_id"],"actual_minutes":labels})
        posted.raise_for_status()
        docker("exec","-T","monitor","python","-c","from food_delivery.watch import tick; tick()")
        windows.append(json.loads((ROOT/"artifacts-container/monitor_live.json").read_text()))
    request = json.loads((ROOT/"artifacts-container/retrain_request.json").read_text())
    assert all(w["label_rows"] == 100 and w["data_drift"] and w["retrain_requested"] for w in windows)
    assert set(windows[0]["label_request_ids"]).isdisjoint(windows[1]["label_request_ids"])
    write_json(OUT/"container_feedback_drift.json",{"scope":"real API feedback and monitor execution, deliberately synthetic +12 minute labels and coordinate drift", "windows":windows,"retrain_request":request})
    print("Two independent synthetic feedback windows triggered retraining request",flush=True)


def wait_alerts(names, filename, timeout=450):
    deadline = time.monotonic()+timeout
    history = []
    last = None
    while time.monotonic() < deadline:
        response = httpx.get(PROM+"/api/v1/alerts").json()
        states = {a["labels"]["alertname"]:a["state"] for a in response["data"]["alerts"]}
        if states != last:
            print(json.dumps(states),flush=True)
            history.append({"time":time.time(),"states":states})
            last = states
        if all(states.get(name) == "firing" for name in names):
            write_json(OUT/filename,{"passed":True,"alerts":response,"history":history,"scope":"actual Prometheus firing state; outbound notification receiver not configured"})
            return
        time.sleep(5)
    write_json(OUT/filename,{"passed":False,"history":history})
    raise RuntimeError("Expected Prometheus alerts did not fire")


def cycle():
    docker("stop","monitor")
    try:
        stages = {}
        for action in ["baseline","retrain","rollback"]:
            docker("exec","-T","api","python","-m","food_delivery.container_demo",action)
            docker("restart","api")
            ready()
            stages[action] = prediction()
        a = np.asarray(stages["baseline"]["predictions_minutes"])
        b = np.asarray(stages["retrain"]["predictions_minutes"])
        c = np.asarray(stages["rollback"]["predictions_minutes"])
        assert stages["baseline"]["model"]["version"] == 1
        assert stages["retrain"]["model"]["version"] == 2
        assert stages["rollback"]["model"]["version"] == "1"
        assert np.mean(b-a) > 8
        np.testing.assert_allclose(a,c,rtol=1e-8,atol=1e-8)
        write_json(OUT/"container_live_rollback.json",{"passed":True,"stages":stages,"prediction_change_minutes":float(np.mean(b-a)),"scope":"actual container HTTP and isolated synthetic registry, simulated +12 minute target shift; original live model restored afterwards"})
    finally:
        docker("exec","-T","api","python","-m","food_delivery.container_demo","restore")
        docker("restart","api")
        ready()
        docker("start","monitor")


def system_alert():
    docker("stop","api")
    try:
        wait_alerts(["DeliveryAPIDown"],"container_system_alert.json",timeout=180)
    finally:
        docker("start","api")
        ready()


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("phase",choices=["baseline","feedback","alerts","cycle","system-alert"])
    phase = parser.parse_args().phase
    {"baseline":baseline,"feedback":feedback,"alerts":lambda:wait_alerts(["DeliveryDataDrift","DeliveryQualityDegraded"],"container_drift_alerts.json"),"cycle":cycle,"system-alert":system_alert}[phase]()
