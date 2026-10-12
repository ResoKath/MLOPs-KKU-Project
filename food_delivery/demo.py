"""Simulate drift, retrain in an isolated registry, and demonstrate rollback."""
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from sklearn.ensemble import HistGradientBoostingRegressor

from food_delivery.model import DeliveryModel
from food_delivery.monitor import evaluate
from food_delivery.preprocessing import TARGET, Preprocessor, prepare_data
from food_delivery.train import metrics, write_json


def demo(data="input/Zomato Dataset.csv", out="artifacts"):
    out = Path(out).resolve()
    (train,val,_),_ = prepare_data(data)
    current = val.sample(500,random_state=17)
    model = joblib.load(out/"model.joblib")
    x = model.preprocessor.transform(current)
    reference = pd.read_csv(out/"reference_features.csv")
    pred = model.predict(current)
    selection = json.loads((out/"selection.json").read_text())
    ref_mae = selection["validation"][selection["selected"]]["mae"]
    healthy = evaluate(reference,x,pred,current[TARGET].values,ref_mae)
    data_shift = x.copy()
    data_shift["distance_km"] = 40.0
    data_drift = evaluate(reference,data_shift)
    shifted_labels = current[TARGET].values + 12
    concept = evaluate(reference,x,pred,shifted_labels,ref_mae)
    assert data_drift["data_drift"] and concept["retrain_requested"]
    # Simulation only: synthesize new labels on training and validation, never alter original files.
    new_train = train.copy()
    new_train[TARGET] += 12
    proc = Preprocessor().fit(new_train)
    sample = new_train.sample(5000,random_state=42).sort_index()
    estimator = HistGradientBoostingRegressor(categorical_features="from_dtype",max_iter=200,random_state=42)
    estimator.fit(proc.transform(sample),sample[TARGET])
    retrained = DeliveryModel(proc,estimator,{"name":"simulated_concept_retrain"})
    after = metrics(shifted_labels,retrained.predict(current))
    before = metrics(shifted_labels,pred)
    assert after["mae"] < before["mae"] and after["mae"] <= 5
    mlflow.set_tracking_uri(f"sqlite:///{(out/'mlflow.db').as_posix()}")
    mlflow.set_experiment("Food_Delivery_CP413008")
    # Isolated demo registry. Never promote synthetic labels to the main model.
    with mlflow.start_run(run_name="simulated_concept_retraining"):
        mlflow.log_dict({"scenario":"labels +12 minutes; synthetic validation only; no production test"},"scenario.json")
        mlflow.log_dict(concept,"monitor.json")
        mlflow.log_dict(proc.config,"preprocessing_config.json")
        mlflow.log_metrics({"before_mae":before["mae"],"after_mae":after["mae"]})
        first = mlflow.sklearn.log_model(model,name="old_demo",serialization_format="cloudpickle",code_paths=["food_delivery"],pip_requirements=[])
        second = mlflow.sklearn.log_model(retrained,name="new_demo",serialization_format="cloudpickle",code_paths=["food_delivery"],pip_requirements=[])
    name = "food-delivery-synthetic-rollback-demo"
    v1 = mlflow.register_model(first.model_uri,name)
    v2 = mlflow.register_model(second.model_uri,name)
    client = MlflowClient()
    client.set_registered_model_alias(name,"champion",v1.version)
    old = mlflow.sklearn.load_model(f"models:/{name}@champion").predict(current.head(10))
    client.set_registered_model_alias(name,"champion",v2.version)
    changed = mlflow.sklearn.load_model(f"models:/{name}@champion").predict(current.head(10))
    assert np.mean(changed-old) > 8
    client.set_registered_model_alias(name,"champion",v1.version)
    reverted = mlflow.sklearn.load_model(f"models:/{name}@champion").predict(current.head(10))
    np.testing.assert_allclose(old,reverted,rtol=1e-8)
    result = {"healthy":healthy,"data_shift":data_drift,"label_shift":concept,
              "retrain_before":before,"retrain_after":after,
              "rollback":{"status":"PASSED","registry":name,"versions":[v1.version,v2.version],
                          "sequence":[v1.version,v2.version,v1.version],"prediction_equivalence":"PASSED"},
              "scope":"synthetic +12 minute labels; isolated registry; no new real observations or production test"}
    write_json(out/"drift_retrain_rollback_demo.json",result)
    return result


if __name__ == "__main__":
    print(json.dumps(demo(),indent=2))
