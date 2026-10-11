"""Leakage-safe experiments, full held-out test, MLflow and quality gates."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from food_delivery.model import DeliveryModel
from food_delivery.preprocessing import CATEGORIES, FEATURES, REQUIRED, TARGET, Preprocessor, prepare_data


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, default=str, allow_nan=False), encoding="utf8")


def code_hash():
    digest = hashlib.sha256()
    for p in sorted(Path("food_delivery").glob("*.py")):
        digest.update(p.name.encode())
        digest.update(p.read_bytes())
    return digest.hexdigest()


def metrics(y, prediction):
    return {"mae": float(mean_absolute_error(y, prediction)),
            "rmse": float(np.sqrt(mean_squared_error(y, prediction))),
            "r2": float(r2_score(y, prediction)),
            "within_5_minutes": float(np.mean(np.abs(np.asarray(y)-prediction) <= 5))}


def gate(validation, dummy_mae, test=None, max_mae=5.0):
    passed = np.isfinite(validation["mae"]) and validation["mae"] <= max_mae and validation["mae"] < dummy_mae
    if test is not None:
        passed = passed and np.isfinite(test["mae"]) and test["mae"] <= max_mae
    return bool(passed)


def estimators(with_tabpfn=False, device="cuda"):
    encoding = ColumnTransformer([
        ("categories", OneHotEncoder(handle_unknown="ignore", sparse_output=False), list(CATEGORIES)),
        ("numbers", "passthrough", [c for c in FEATURES if c not in CATEGORIES]),
    ])
    models = {"dummy": DummyRegressor(strategy="mean"),
              "linear": Pipeline([("encode", encoding), ("model", LinearRegression())]),
              "hist_gradient_boosting": HistGradientBoostingRegressor(
                  categorical_features="from_dtype", max_iter=200, learning_rate=0.1,
                  max_leaf_nodes=31, random_state=42)}
    if with_tabpfn:
        from tabpfn import TabPFNRegressor
        models["tabpfn"] = TabPFNRegressor(device=device, n_estimators=1, fit_mode="low_memory", random_state=42)
    return models


def batched_predict(model, raw, batch_size=100):
    return np.concatenate([model.predict(raw.iloc[i:i+batch_size]) for i in range(0, len(raw), batch_size)])


def run(data_path, out="artifacts", with_tabpfn=False, device="cuda", train_rows=5000, max_mae=5.0):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    uri = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{(out/'mlflow.db').as_posix()}")
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    client = MlflowClient()
    experiment_name = "Food_Delivery_CP413008"
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        client.create_experiment(experiment_name, artifact_location=(out/"mlruns").as_uri())
    mlflow.set_experiment(experiment_name)
    (train, val, test), quality = prepare_data(data_path)
    write_json(out/"data_quality.json", quality)
    # Fit medians on complete train split, never on validation/test.
    processor = Preprocessor().fit(train)
    processor.save(out/"preprocessing_config.json")
    restored = Preprocessor.load(out/"preprocessing_config.json")
    for label, part in [("train", train), ("validation", val), ("test", test)]:
        pd.testing.assert_frame_equal(processor.transform(part), restored.transform(part))
    write_json(out/"consistency.json", {"status": "PASSED", "scope": "all three splits; exact JSON reload; shared training/serving transform"})
    # Seeded sample covers the whole training date range (unlike the original first 5000 rows).
    sample = train.sample(n=min(train_rows, len(train)), random_state=42).sort_index()
    x = processor.transform(sample)
    y = sample[TARGET]
    env = {p: importlib.metadata.version(p) for p in ["pandas", "numpy", "scikit-learn", "mlflow", "fastapi", "joblib"]}
    if with_tabpfn:
        env["tabpfn"] = importlib.metadata.version("tabpfn")
    try:
        commit = subprocess.check_output([os.environ.get("GIT_EXECUTABLE","git"), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable; use source_sha256"
    provenance = {"data_sha256": hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
                  "source_sha256": code_hash(), "git_commit": commit,
                  "python": platform.python_version(), "platform": platform.platform(), "libraries": env,
                  "training_policy": "seeded random train-only sample; 5000 default; frozen before test",
                  "test_policy": "full test evaluated only for validation-selected winner and TabPFN audit; never tune on test"}
    write_json(out/"environment.json", provenance)
    results, trained, run_ids = {}, {}, {}
    for name, estimator in estimators(with_tabpfn, device).items():
        with mlflow.start_run(run_name=name) as active:
            mlflow.log_params({"model": name, "train_rows": len(sample), "validation_rows": len(val),
                               "random_state": 42, "data_sha256": provenance["data_sha256"],
                               "source_sha256": provenance["source_sha256"]})
            mlflow.log_dict(estimator.get_params(deep=False), "hyperparameters.json")
            mlflow.log_dict(provenance, "environment.json")
            mlflow.log_dict(quality, "data_quality.json")
            mlflow.log_artifact(str(out/"preprocessing_config.json"))
            start = time.perf_counter()
            estimator.fit(x, y)
            bundle = DeliveryModel(processor, estimator, {"name": name, "run_id": active.info.run_id})
            prediction = batched_predict(bundle, val)
            result = metrics(val[TARGET], prediction)
            result["fit_validation_seconds"] = time.perf_counter()-start
            mlflow.log_metrics({f"validation_{k}": v for k,v in result.items()})
            pred_path = out/f"{name}_validation_predictions.csv"
            pd.DataFrame({"ID": val.ID, "actual": val[TARGET], "prediction": prediction}).to_csv(pred_path, index=False)
            mlflow.log_artifact(str(pred_path))
            results[name], trained[name], run_ids[name] = result, bundle, active.info.run_id
    selected = min((n for n in results if n != "dummy"), key=lambda n: results[n]["mae"])
    # Freeze choice before exposing test metrics.
    write_json(out/"selection.json", {"selected": selected, "criterion": "lowest full validation MAE", "validation": results})
    if not gate(results[selected], results["dummy"]["mae"], max_mae=max_mae):
        write_json(out/"gate.json", {"passed": False, "reason": "validation gate", "selected": selected})
        raise ValueError("Validation quality gate failed; test and deployment blocked")
    test_results = {}
    for name in sorted(set([selected] + (["tabpfn"] if with_tabpfn else []))):
        prediction = batched_predict(trained[name], test)
        test_results[name] = metrics(test[TARGET], prediction)
        path = out/f"{name}_test_predictions.csv"
        pd.DataFrame({"ID": test.ID, "actual": test[TARGET], "prediction": prediction}).to_csv(path, index=False)
        with mlflow.start_run(run_id=run_ids[name]):
            mlflow.log_metrics({f"test_{k}":v for k,v in test_results[name].items()})
            mlflow.log_artifact(str(path))
    write_json(out/"test_metrics.json", test_results)
    if not gate(results[selected], results["dummy"]["mae"], test_results[selected], max_mae):
        write_json(out/"gate.json", {"passed": False, "reason": "test safety gate; do not retune on test"})
        raise ValueError("Held-out test gate failed; registration blocked")
    bundle = trained[selected]
    raw_example = val[REQUIRED].head(5).copy()
    with mlflow.start_run(run_id=run_ids[selected]):
        info = mlflow.sklearn.log_model(bundle, name="raw_delivery_model", serialization_format="cloudpickle",
            signature=infer_signature(raw_example, bundle.predict(raw_example)), input_example=raw_example,
            code_paths=["food_delivery"], pip_requirements=[f"{k}=={v}" for k,v in env.items()])
        reloaded = mlflow.sklearn.load_model(info.model_uri)
        np.testing.assert_allclose(bundle.predict(raw_example), reloaded.predict(raw_example), rtol=1e-8)
    registry_name = f"food-delivery-{selected}"
    version = mlflow.register_model(info.model_uri, registry_name)
    client.set_model_version_tag(registry_name, version.version, "approval_status", "candidate_quality_passed")
    client.set_registered_model_alias(registry_name, "candidate", version.version)
    registry_model = mlflow.sklearn.load_model(f"models:/{registry_name}/{version.version}")
    np.testing.assert_allclose(bundle.predict(raw_example), registry_model.predict(raw_example), rtol=1e-8)
    joblib.dump(bundle, out/"candidate.joblib")
    raw_example.astype(object).where(raw_example.notna(), None).to_json(out/"example_request_records.json", orient="records", indent=2)
    decision = {"passed": True, "max_mae": max_mae, "registry_name": registry_name, "version": version.version,
                "model_uri": info.model_uri, "registry_reload": "PASSED", "selected": selected,
                "status": "candidate; API latency/artifact gates required before approval"}
    write_json(out/"gate.json", decision)
    write_json(out/"registry.json", decision)
    # Reference only uses training features; no test data in monitoring reference.
    processor.transform(train).to_csv(out/"reference_features.csv", index=False)
    print(json.dumps({"selected": selected, "validation": results, "test": test_results, "registry": decision}, indent=2))
    return decision


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="input/Zomato Dataset.csv")
    parser.add_argument("--out", default="artifacts")
    parser.add_argument("--tabpfn", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train-rows", type=int, default=5000)
    parser.add_argument("--max-mae", type=float, default=5.0)
    args = parser.parse_args()
    run(args.data, args.out, args.tabpfn, args.device, args.train_rows, args.max_mae)
