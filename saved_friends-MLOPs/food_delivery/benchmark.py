"""Measure HTTP latency and throughput; includes network/validation/log writes."""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np


def benchmark(url="http://127.0.0.1:8000", records_path="artifacts/example_request_records.json", out="artifacts/benchmark.json", count=100, concurrency=4):
    payload = {"records": json.loads(Path(records_path).read_text(encoding="utf8"))[:1]}
    with httpx.Client(timeout=30) as client:
        client.get(url+"/health/ready").raise_for_status()
        for _ in range(10):
            client.post(url+"/predict", json=payload).raise_for_status()

        def call(_):
            start = time.perf_counter()
            response = client.post(url+"/predict", json=payload)
            return time.perf_counter()-start, response.status_code

        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            responses = list(pool.map(call, range(count)))
        elapsed = time.perf_counter()-start
        latencies = [r[0] for r in responses]
        errors = sum(r[1] != 200 for r in responses)
        result = {"samples": count, "concurrency": concurrency, "warmup_requests": 10,
                  "p50_seconds": float(np.percentile(latencies,50)),
                  "p95_seconds": float(np.percentile(latencies,95)),
                  "throughput_requests_per_second": (count-errors)/elapsed,
                  "error_rate": errors/count, "elapsed_seconds": elapsed,
                  "scope": "localhost HTTP; CPU; one record/request; not container or production load"}
        result["slo"] = {"p95_seconds_max": 1.0, "throughput_min": 10.0, "error_rate_max": 0.01}
        result["passed"] = result["p95_seconds"] <= 1 and result["throughput_requests_per_second"] >= 10 and result["error_rate"] <= .01
        Path(out).write_text(json.dumps(result, indent=2), encoding="utf8")
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    print(json.dumps(benchmark(url=args.url), indent=2))
