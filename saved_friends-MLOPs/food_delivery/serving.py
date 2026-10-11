import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

from food_delivery.benchmark import benchmark
from food_delivery.train import write_json


def serving_gate(output):
    output = Path(output).resolve()
    env = os.environ.copy()
    env["MODEL_PATH"] = str(output/"candidate.joblib")
    env["EVENT_DB"] = str(output/"events.db")
    log = (output/"api-smoke.log").open("w", encoding="utf8")
    process = subprocess.Popen([sys.executable,"-m","uvicorn","food_delivery.api:app","--host","127.0.0.1","--port","8765"],
                               env=env,stdout=log,stderr=subprocess.STDOUT,
                               creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    try:
        for _ in range(60):
            if process.poll() is not None:
                raise RuntimeError("API failed to start; inspect api-smoke.log")
            try:
                response = httpx.get("http://127.0.0.1:8765/health/ready",timeout=1)
                if response.status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(.5)
        else:
            raise RuntimeError("API readiness timeout")
        measured = benchmark("http://127.0.0.1:8765",str(output/"example_request_records.json"),str(output/"benchmark.json"))
        invalid = httpx.post("http://127.0.0.1:8765/predict",json={"records":[{}]})
        write_json(output/"bad_request_demo.json",{"status_code":invalid.status_code,"response":invalid.json()})
        if invalid.status_code != 422 or not measured["passed"]:
            raise ValueError("HTTP serving gate failed")
        return measured
    finally:
        process.terminate()
        process.wait(timeout=15)
        log.close()


