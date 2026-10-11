import hashlib
import json
import sqlite3
from types import SimpleNamespace

import joblib
import pandas as pd
import pytest

from food_delivery.evidence import identity
from food_delivery.monitor import from_events
from food_delivery.preprocessing import Preprocessor, prepare_data
from food_delivery.train import incumbent_gate


def test_incumbent_gate_rejects_regression():
    assert incumbent_gate(3,4)
    assert incumbent_gate(3,3)
    assert not incumbent_gate(4,3)
    assert not incumbent_gate(float("nan"),3)


def test_changed_candidate_rejects_stale_evidence(tmp_path):
    joblib.dump(SimpleNamespace(metadata={"run_id":"new"}), tmp_path/"candidate.joblib")
    decision = {"run_id":"new", "registry_name":"test", "version":"1", "data_sha256":"data",
                "candidate_sha256":hashlib.sha256((tmp_path/"candidate.joblib").read_bytes()).hexdigest()}
    (tmp_path/"gate.json").write_text(json.dumps(decision))
    assert identity(tmp_path)["version"] == "1"
    joblib.dump(SimpleNamespace(metadata={"run_id":"new","changed":True}), tmp_path/"candidate.joblib")
    with pytest.raises(ValueError, match="hash"):
        identity(tmp_path)


def test_alias_failure_restores_live_model(monkeypatch,tmp_path):
    from food_delivery import registry
    joblib.dump(SimpleNamespace(metadata={"run_id":"run"}),tmp_path/"candidate.joblib")
    decision = {"passed":True,"run_id":"run","registry_name":"test","version":"2","data_sha256":"data",
                "candidate_sha256":hashlib.sha256((tmp_path/"candidate.joblib").read_bytes()).hexdigest()}
    (tmp_path/"gate.json").write_text(json.dumps(decision))
    (tmp_path/"benchmark.json").write_text(json.dumps({"passed":True,"identity":{key:decision[key] for key in ("run_id","registry_name","version","candidate_sha256","data_sha256")}}))
    (tmp_path/"model.joblib").write_bytes(b"previous model")
    class FailingClient:
        def get_model_version(self,*args):
            return SimpleNamespace(run_id="run")
        def set_registered_model_alias(self,*args):
            raise RuntimeError("registry unavailable")
    monkeypatch.setattr(registry,"MlflowClient",FailingClient)
    monkeypatch.setattr(registry.mlflow,"set_tracking_uri",lambda _:None)
    with pytest.raises(RuntimeError,match="registry unavailable"):
        registry.approve(tmp_path)
    assert (tmp_path/"model.joblib").read_bytes() == b"previous model"


def test_monitor_only_active_version_and_unconsumed_labels(tmp_path):
    (train, _, _), _ = prepare_data("input/Zomato Dataset.csv")
    raw = train.head(5)
    features = Preprocessor().fit(raw).transform(pd.concat([raw]*20, ignore_index=True))
    features.to_csv(tmp_path/"reference.csv",index=False)
    with sqlite3.connect(tmp_path/"events.db") as db:
        db.execute("CREATE TABLE events(request_id TEXT,created REAL,model_version TEXT,features TEXT,predictions TEXT,actual TEXT)")
        for rid, version, actual in [("old","1",100),("new","2",20)]:
            db.execute("INSERT INTO events VALUES(?,?,?,?,?,?)",(rid,1,version,features.to_json(orient="records"),json.dumps([20]*100),json.dumps([actual]*100)))
    result = from_events(tmp_path/"reference.csv",tmp_path/"events.db",tmp_path/"result.json",3,model_version="2")
    assert result["label_rows"] == 100 and result["mae"] == 0
    assert result["model_version"] == "2" and result["label_request_ids"] == ["new"]
    second = from_events(tmp_path/"reference.csv",tmp_path/"events.db",tmp_path/"result.json",3,model_version="2",consumed_labels=["new"])
    assert second["label_rows"] == 0 and second["quality_status"] == "labels_unavailable"
