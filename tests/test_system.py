import json

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.dummy import DummyRegressor

from food_delivery.api import create_app
from food_delivery.model import DeliveryModel
from food_delivery.monitor import evaluate
from food_delivery.preprocessing import FEATURES, Preprocessor, parse_time, prepare_data, validate_raw
from food_delivery.train import gate


@pytest.fixture
def raw():
    return pd.DataFrame([{
        "Delivery_person_Age": 30.0, "Delivery_person_Ratings":4.5,
        "Restaurant_latitude":18.52,"Restaurant_longitude":73.85,
        "Delivery_location_latitude":18.55,"Delivery_location_longitude":73.88,
        "Weather_conditions":"Sunny","Road_traffic_density":"Low","Vehicle_condition":1,
        "Type_of_order":"Meal","Type_of_vehicle":"motorcycle","Festival":"No","City":"Urban",
        "Time_Orderd":"12:00"}, {
        "Delivery_person_Age": 40.0, "Delivery_person_Ratings":4.7,
        "Restaurant_latitude":18.52,"Restaurant_longitude":73.85,
        "Delivery_location_latitude":18.55,"Delivery_location_longitude":73.88,
        "Weather_conditions":"Fog","Road_traffic_density":"High","Vehicle_condition":2,
        "Type_of_order":"Snack","Type_of_vehicle":"scooter","Festival":"Yes","City":"Metropolitian",
        "Time_Orderd":"18:00"}])


def test_train_only_medians_json_consistency(raw,tmp_path):
    proc = Preprocessor().fit(raw)
    missing = raw.copy()
    missing.loc[0,"Delivery_person_Age"] = 100
    missing.loc[1,"Time_Orderd"] = None
    proc.save(tmp_path/"config.json")
    loaded = Preprocessor.load(tmp_path/"config.json")
    pd.testing.assert_frame_equal(proc.transform(missing),loaded.transform(missing))
    assert loaded.transform(missing).loc[0,"Delivery_person_Age"] == 35
    assert loaded.transform(missing).loc[1,"order_time_missing"] == 1
    assert list(proc.transform(raw)) == FEATURES
    assert proc.config["medians"] == loaded.config["medians"]


@pytest.mark.parametrize("column,value",[("Restaurant_latitude",0),("Vehicle_condition",9),
    ("Weather_conditions","Hurricane"),("Time_Orderd","bad"),("Delivery_person_Age",float("inf"))])
def test_corrupt_input_blocks(raw,column,value):
    raw.loc[0,column] = value
    with pytest.raises(ValueError):
        validate_raw(raw)


def test_missing_required_blocks(raw):
    with pytest.raises(ValueError,match="Missing columns"):
        validate_raw(raw.drop(columns="Type_of_vehicle"))


def test_excel_fraction_times():
    t = parse_time(pd.Series(["0.833333333","1","09:20",None]))
    assert t.iloc[0].hour == 20
    assert t.iloc[1].hour == 0
    assert t.iloc[2].minute == 20
    assert pd.isna(t.iloc[3])


def test_temporal_split_real_data():
    parts,quality = prepare_data("input/Zomato Dataset.csv")
    assert [len(p) for p in parts] == [29674,5910,5929]
    assert quality["invalid_gps_rows"] == 4071
    assert parts[0].Order_Date.max() < parts[1].Order_Date.min()
    assert parts[1].Order_Date.max() < parts[2].Order_Date.min()
    assert not set(parts[0].ID) & set(parts[2].ID)


def test_gates_accept_reject():
    assert gate({"mae":3},8,{"mae":4})
    assert not gate({"mae":6},8)
    assert not gate({"mae":3},8,{"mae":6})
    assert not gate({"mae":float("nan")},8)


def test_api_roundtrip_feedback_and_failures(raw,tmp_path):
    proc = Preprocessor().fit(raw)
    est = DummyRegressor().fit(proc.transform(raw),[20,30])
    bundle = DeliveryModel(proc,est)
    model_path = tmp_path/"model.joblib"
    joblib.dump(bundle,model_path)
    with TestClient(create_app(model_path,tmp_path/"events.db")) as client:
        assert client.get("/health/ready").status_code == 200
        payload = {"records":raw.to_dict("records")}
        response = client.post("/predict",json=payload)
        assert response.status_code == 200
        assert response.json()["predictions_minutes"] == bundle.predict(raw).tolist()
        rid = response.json()["request_id"]
        assert client.post("/feedback",json={"request_id":rid,"actual_minutes":[21]}).status_code == 422
        assert client.post("/feedback",json={"request_id":rid,"actual_minutes":[21,31]}).status_code == 200
        assert client.post("/feedback",json={"request_id":rid,"actual_minutes":[21,31]}).status_code == 409
        assert client.post("/predict",json={"records":[{}]}).status_code == 422
        assert client.post("/predict",json={"records":[]}).status_code == 422
        assert client.post("/predict",json={"records":raw.to_dict("records")*51}).status_code == 422
        assert 'status="invalid"' in client.get("/metrics").text


def test_api_unavailable(tmp_path):
    with TestClient(create_app(tmp_path/"absent",tmp_path/"events.db")) as client:
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").status_code == 503
        assert client.post("/predict",json={"records":[{}]}).status_code == 503


def test_drift_detection_with_delayed_labels(raw):
    x = Preprocessor().fit(raw).transform(pd.concat([raw]*100,ignore_index=True))
    healthy = evaluate(x,x,np.full(200,25),np.full(200,25),3)
    assert not healthy["data_drift"] and not healthy["retrain_requested"]
    changed = x.copy()
    changed["distance_km"] = 40
    assert evaluate(x,changed)["data_drift"]
    label_changed = evaluate(x,x,np.full(200,25),np.full(200,40),3)
    assert not label_changed["data_drift"]
    assert label_changed["concept_drift_signal"] and label_changed["retrain_requested"]
    assert evaluate(x,x.head(5))["status"] == "insufficient_data"


def test_trained_artifact_real_raw_prediction():
    path = "artifacts/model.joblib"
    from pathlib import Path
    if not Path(path).exists():
        pytest.skip("run training first")
    bundle = joblib.load(path)
    records = json.loads(Path("artifacts/example_request_records.json").read_text())
    predictions = bundle.predict(pd.DataFrame(records))
    assert np.isfinite(predictions).all() and (predictions > 0).all()


def test_local_dag_stops_before_training(monkeypatch,tmp_path):
    from food_delivery import pipeline

    def reject(_):
        raise ValueError("Corrupt input blocked")

    monkeypatch.setattr(pipeline.validate,"fn",reject)
    with pytest.raises(ValueError,match="Corrupt input blocked"):
        pipeline.local_dag("bad.csv",str(tmp_path))
    state = json.loads((tmp_path/"dag_state.json").read_text())
    assert state["states"]["validate"]["status"] == "FAILED"
    assert "train_evaluate_register" not in state["states"]
