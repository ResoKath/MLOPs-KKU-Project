"""Realtime API with persistent prediction/feedback events and Prometheus metrics."""
import json
import os
import sqlite3
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    records: list[dict] = Field(min_length=1, max_length=100)


class Feedback(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    request_id: str
    actual_minutes: list[float] = Field(min_length=1, max_length=100)


def create_app(model_path=None, log_path=None):
    model_path = Path(model_path or os.environ.get("MODEL_PATH", "artifacts/model.joblib"))
    log_path = Path(log_path or os.environ.get("EVENT_DB", "artifacts/events.db"))
    lock = threading.Lock()
    counters = {"success": 0, "invalid": 0, "error": 0, "seconds": 0.0}
    buckets = {bound: 0 for bound in [.05,.1,.25,.5,1.,2.,5.]}

    @asynccontextmanager
    async def lifespan(app):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(log_path) as db:
            db.execute("CREATE TABLE IF NOT EXISTS events (request_id TEXT PRIMARY KEY, created REAL, model_version TEXT, predictions TEXT, features TEXT, actual TEXT)")
        app.state.model = joblib.load(model_path) if model_path.exists() else None
        yield

    app = FastAPI(title="Food Delivery Time Prediction", version="1.0.0", lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def invalid_payload(request, exc):
        if request.url.path == "/predict":
            with lock:
                counters["invalid"] += 1
        return await request_validation_exception_handler(request, exc)

    @app.get("/health/live")
    def live():
        return {"status": "alive"}

    @app.get("/health/ready")
    def ready():
        if app.state.model is None:
            raise HTTPException(503, "Model unavailable")
        return {"status": "ready", "model": app.state.model.metadata}

    @app.post("/predict")
    def predict(request: PredictionRequest):
        if app.state.model is None:
            raise HTTPException(503, "Model unavailable")
        start = time.perf_counter()
        try:
            raw = pd.DataFrame(request.records)
            with lock:
                features = app.state.model.preprocessor.transform(raw)
                predictions = app.state.model.predict(raw).tolist()
            request_id = str(uuid.uuid4())
            # Store derived features, never exact GPS, rider/order IDs, or tokens.
            with sqlite3.connect(log_path, timeout=10) as db:
                db.execute("INSERT INTO events VALUES (?,?,?,?,?,NULL)",
                    (request_id, time.time(), str(app.state.model.metadata.get("version", "candidate")),
                     json.dumps(predictions), features.to_json(orient="records")))
            elapsed = time.perf_counter()-start
            with lock:
                counters["success"] += 1
                counters["seconds"] += elapsed
                for bound in buckets:
                    buckets[bound] += int(elapsed <= bound)
            return {"request_id": request_id, "predictions_minutes": predictions,
                    "model": app.state.model.metadata, "processing_seconds": elapsed}
        except ValueError as error:
            with lock:
                counters["invalid"] += 1
            raise HTTPException(422, str(error)) from error
        except Exception as error:
            with lock:
                counters["error"] += 1
            raise HTTPException(500, "Prediction or event persistence failed") from error

    @app.post("/feedback")
    def feedback(request: Feedback):
        if any(x <= 0 for x in request.actual_minutes):
            raise HTTPException(422, "actual_minutes must be positive")
        with sqlite3.connect(log_path) as db:
            row = db.execute("SELECT predictions,actual FROM events WHERE request_id=?", (request.request_id,)).fetchone()
            if row is None:
                raise HTTPException(404, "Unknown request_id")
            if row[1] is not None:
                raise HTTPException(409, "Feedback already recorded")
            if len(json.loads(row[0])) != len(request.actual_minutes):
                raise HTTPException(422, "Feedback length differs from predictions")
            db.execute("UPDATE events SET actual=? WHERE request_id=? AND actual IS NULL",
                       (json.dumps(request.actual_minutes), request.request_id))
        return {"status": "recorded"}

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics():
        with lock:
            values = counters.copy()
            bucket_values = buckets.copy()
        quality = ""
        monitor_path = model_path.parent/"monitor_live.json"
        if monitor_path.exists():
            report = json.loads(monitor_path.read_text())
            quality = f"delivery_data_drift {int(report.get('data_drift',False))}\n"
            quality += f"delivery_quality_degraded {int(report.get('retrain_requested',False))}\n"
            if "mae" in report:
                quality += f"delivery_label_mae_minutes {report['mae']}\n"
        return ("# TYPE delivery_requests_total counter\n" +
                "".join(f'delivery_requests_total{{status="{k}"}} {values[k]}\n' for k in ["success", "invalid", "error"]) +
                "# TYPE delivery_processing_seconds histogram\n" +
                "".join(f'delivery_processing_seconds_bucket{{le="{bound}"}} {value}\n' for bound,value in bucket_values.items()) +
                f'delivery_processing_seconds_bucket{{le="+Inf"}} {values["success"]}\n' +
                f"delivery_processing_seconds_sum {values['seconds']}\n" +
                f"delivery_processing_seconds_count {values['success']}\n" +
                f"delivery_model_ready {int(app.state.model is not None)}\n" + quality)

    return app


app = create_app()
