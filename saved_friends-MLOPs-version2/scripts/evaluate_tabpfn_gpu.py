"""Frozen new TabPFN experiment: shared preprocessing, full validation and test."""
# Cache settings and project import path must be set before TabPFN is imported.
# ruff: noqa: E402
import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TABPFN_MODEL_CACHE_DIR", str(ROOT / ".cache-tabpfn/models"))
os.environ.setdefault("TABPFN_NO_BROWSER", "1")
os.environ.setdefault("HF_HOME", str(ROOT / ".cache-tabpfn/huggingface"))
os.environ.setdefault("TABPFN_SERVING_DEVICE", "cuda")
os.environ.setdefault("SKB_DATA_DIRECTORY", str(ROOT / ".cache-tabpfn/skrub"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache-tabpfn/matplotlib"))
os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache-tabpfn/torch"))

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
import torch
from mlflow import MlflowClient
from mlflow.models import infer_signature
from tabpfn import TabPFNRegressor
from tabpfn.model_loading import save_fitted_tabpfn_model

from food_delivery.preprocessing import REQUIRED, TARGET, Preprocessor, prepare_data
from food_delivery.tabpfn_model import RawTabPFNModel
from food_delivery.train import code_hash, metrics, write_json


def predict_batches(estimator, features, batch_size, label, out):
    predictions = []
    started = time.perf_counter()
    for offset in range(0, len(features), batch_size):
        predictions.extend(estimator.predict(features.iloc[offset:offset + batch_size]))
        if offset == 0 or offset // 500 != (offset + batch_size) // 500 or offset + batch_size >= len(features):
            done = min(offset + batch_size, len(features))
            state = {"stage": label, "completed_rows": done, "total_rows": len(features),
                     "seconds": time.perf_counter() - started}
            write_json(out / "progress.json", state)
            print(f"{label}: {done}/{len(features)} rows; {state['seconds']:.1f}s", flush=True)
    result = np.asarray(predictions, dtype=float)
    if len(result) != len(features) or not np.isfinite(result).all():
        raise ValueError("Incomplete or non-finite predictions")
    return result, time.perf_counter() - started


def run(batch_size=16, checkpoint=".cache-tabpfn/models/tabpfn-v3.5-20260909.safetensors"):
    out = ROOT / "artifacts-tabpfn-gpu"
    out.mkdir(exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA tensor verification must pass before evaluation")
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.exists():
        raise FileNotFoundError("Official checkpoint must be downloaded first")
    uri = f"sqlite:///{(out / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    client = MlflowClient()
    name = "Food_Delivery_TabPFN_GPU"
    if client.get_experiment_by_name(name) is None:
        client.create_experiment(name, artifact_location=(out / "mlruns").as_uri())
    mlflow.set_experiment(name)
    (train, validation, test), quality = prepare_data(ROOT / "input/Zomato Dataset.csv")
    processor = Preprocessor().fit(train)
    processor.save(out / "preprocessing_config.json")
    recovered = Preprocessor.load(out / "preprocessing_config.json")
    for part in [train, validation, test]:
        pd.testing.assert_frame_equal(processor.transform(part), recovered.transform(part))
    sample = train.sample(5000, random_state=42).sort_index()
    sample[["ID"]].to_csv(out / "training_sample_ids.csv", index=False)
    write_json(out / "data_quality.json", quality)
    write_json(out / "consistency.json", {"status": "PASSED", "splits": ["train", "validation", "test"]})
    x_train = processor.transform(sample)
    x_validation = processor.transform(validation)
    x_test = processor.transform(test)
    estimator = TabPFNRegressor(model_path=str(checkpoint), device="cuda", n_estimators=1,
                               fit_mode="low_memory", memory_saving_mode=True, random_state=42)
    versions = {package: importlib.metadata.version(package) for package in
                ["torch", "tabpfn", "pandas", "numpy", "scikit-learn", "mlflow", "joblib", "safetensors"]}
    provenance = {"python": platform.python_version(), "platform": platform.platform(), "libraries": versions,
                  "gpu": torch.cuda.get_device_name(0), "cuda_runtime": torch.version.cuda,
                  "source_sha256": code_hash(), "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "data_sha256": hashlib.sha256((ROOT / "input/Zomato Dataset.csv").read_bytes()).hexdigest(),
                  "experiment_scope": "new training run; revised Excel-time preprocessing; not original Colab fitted weights",
                  "test_policy": "freeze checkpoint, sample, preprocessing and parameters before test; full chronological test"}
    write_json(out / "environment.json", provenance)
    with mlflow.start_run(run_name="tabpfn-v3.5-5000-cuda") as active:
        mlflow.log_params({"model": "TabPFNRegressor", "checkpoint": checkpoint.name, "train_rows": len(sample),
                           "n_estimators": 1, "fit_mode": "low_memory", "memory_saving_mode": True,
                           "random_state": 42, "batch_size": batch_size,
                           "data_sha256": provenance["data_sha256"], "source_sha256": provenance["source_sha256"]})
        mlflow.log_dict(provenance, "environment.json")
        mlflow.log_dict(quality, "data_quality.json")
        mlflow.log_artifact(str(out / "preprocessing_config.json"))
        mlflow.log_artifact(str(out / "training_sample_ids.csv"))
        print("Fitting train-only sample: 5000 rows", flush=True)
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        estimator.fit(x_train, sample[TARGET])
        mlflow.log_metric("fit_seconds", time.perf_counter() - started)
        val_prediction, val_time = predict_batches(estimator, x_validation, batch_size, "validation", out)
        val_metrics = metrics(validation[TARGET], val_prediction)
        mlflow.log_metrics({"validation_" + key: value for key, value in val_metrics.items()})
        mlflow.log_metric("validation_prediction_seconds", val_time)
        write_json(out / "validation_metrics.json", val_metrics)
        print("Validation complete; parameters now frozen for full test", flush=True)
        write_json(out / "frozen_test_plan.json", {"checkpoint": checkpoint.name, "batch_size": batch_size,
                   "train_rows": len(sample), "test_rows": len(test), "selection_or_tuning_on_test": False})
        test_prediction, test_time = predict_batches(estimator, x_test, batch_size, "test", out)
        test_metrics = metrics(test[TARGET], test_prediction)
        mlflow.log_metrics({"test_" + key: value for key, value in test_metrics.items()})
        mlflow.log_metric("test_prediction_seconds", test_time)
        peak_gpu_bytes = torch.cuda.max_memory_allocated()
        mlflow.log_metric("peak_torch_gpu_allocated_bytes", peak_gpu_bytes)
        write_json(out / "test_metrics.json", test_metrics)
        for label, part, prediction in [("validation", validation, val_prediction), ("test", test, test_prediction)]:
            target = out / (label + "_predictions.csv")
            pd.DataFrame({"ID": part.ID, "actual": part[TARGET], "prediction": prediction}).to_csv(target, index=False)
            mlflow.log_artifact(str(target))
        fit_file = out / "model.tabpfn_fit"
        save_fitted_tabpfn_model(estimator, fit_file)
        sample_raw = validation[REQUIRED].head(5).copy()
        expected = estimator.predict(processor.transform(sample_raw))
        signature = infer_signature(sample_raw, expected)
        del estimator
        gc.collect()
        torch.cuda.empty_cache()
        info = mlflow.pyfunc.log_model(name="raw_tabpfn", python_model=RawTabPFNModel(),
            artifacts={"fitted_state": str(fit_file), "checkpoint": str(checkpoint),
                       "preprocessing": str(out / "preprocessing_config.json")},
            model_config={"batch_size": batch_size}, signature=signature, input_example=sample_raw,
            code_paths=[str(ROOT / "food_delivery")],
            pip_requirements=["--extra-index-url https://download.pytorch.org/whl/cu126"] +
                             [f"{key}=={value}" for key, value in versions.items()])
        gc.collect()
        torch.cuda.empty_cache()
        loaded = mlflow.pyfunc.load_model(info.model_uri)
        np.testing.assert_allclose(expected, loaded.predict(sample_raw), rtol=1e-4, atol=1e-4)
        quality_passed = val_metrics["mae"] <= 5 and test_metrics["mae"] <= 5
        registry_name = "food-delivery-tabpfn-gpu"
        version = mlflow.register_model(info.model_uri, registry_name)
        client.set_model_version_tag(registry_name, version.version, "approval_status",
                                     "quality_passed_not_serving_approved" if quality_passed else "quality_rejected")
        if quality_passed:
            client.set_registered_model_alias(registry_name, "candidate", version.version)
        del loaded
        gc.collect()
        torch.cuda.empty_cache()
        registered = mlflow.pyfunc.load_model(f"models:/{registry_name}/{version.version}")
        np.testing.assert_allclose(expected, registered.predict(sample_raw), rtol=1e-4, atol=1e-4)
        total_size = checkpoint.stat().st_size + fit_file.stat().st_size
        summary = {"status": "EVALUATION_PASSED", "run_id": active.info.run_id, "validation": val_metrics,
                   "test": test_metrics, "test_rows": len(test), "preprocessing_consistency": "PASSED",
                   "logged_model_reload": "PASSED", "registry_reload": "PASSED",
                   "registry_name": registry_name, "version": version.version,
                   "quality_gate_passed": bool(quality_passed), "packaged_model_bytes": total_size,
                   "peak_torch_gpu_allocated_bytes": peak_gpu_bytes,
                   "existing_100MB_size_gate_passed": total_size <= 100 * 1024**2,
                   "serving_gate": "not measured; no production/champion promotion",
                   "scope": provenance["experiment_scope"]}
        write_json(out / "result.json", summary)
        mlflow.log_dict(summary, "result.json")
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--checkpoint", default=".cache-tabpfn/models/tabpfn-v3.5-20260909.safetensors")
    args = parser.parse_args()
    try:
        run(args.batch_size, args.checkpoint)
    except Exception as error:
        write_json(ROOT / "artifacts-tabpfn-gpu/failure.json", {"status": "FAILED", "exception_type": type(error).__name__,
                   "message": str(error).replace(os.environ.get("TABPFN_TOKEN", "__no_secret__"), "[redacted]")})
        raise
