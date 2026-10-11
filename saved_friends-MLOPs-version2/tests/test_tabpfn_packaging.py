"""Native fitted-state relocation must not depend on the training machine path."""
import json
import sys
import types
import zipfile
from types import SimpleNamespace

import numpy as np
import pandas as pd

from food_delivery.tabpfn_model import RawTabPFNModel
from food_delivery.preprocessing import Preprocessor


def test_native_checkpoint_path_relocated_without_changing_original(monkeypatch, tmp_path):
    state = tmp_path / "original.tabpfn_fit"
    with zipfile.ZipFile(state, "w") as archive:
        archive.writestr("init_params.json", json.dumps({"model_path": "old-machine/checkpoint.safetensors"}))
        archive.writestr("fitted_attrs.joblib", b"preserved fitted state")
    calls, loaded_params = [], {}

    class Estimator:
        def predict(self, features):
            calls.append(len(features))
            return np.full(len(features), 27.0)

    def loader(path, device):
        with zipfile.ZipFile(path) as archive:
            loaded_params.update(json.loads(archive.read("init_params.json")))
            assert archive.read("fitted_attrs.joblib") == b"preserved fitted state"
        assert device == "cuda"
        return Estimator()

    package = types.ModuleType("tabpfn")
    package.__path__ = []
    module = types.ModuleType("tabpfn.model_loading")
    module.load_fitted_tabpfn_model = loader
    monkeypatch.setitem(sys.modules, "tabpfn", package)
    monkeypatch.setitem(sys.modules, "tabpfn.model_loading", module)
    monkeypatch.setenv("TABPFN_SERVING_DEVICE", "cuda")
    checkpoint = str(tmp_path / "new-machine.safetensors")
    raw = pd.DataFrame([{
        "Delivery_person_Age": 30.0, "Delivery_person_Ratings": 4.5,
        "Restaurant_latitude": 18.52, "Restaurant_longitude": 73.85,
        "Delivery_location_latitude": 18.55, "Delivery_location_longitude": 73.88,
        "Weather_conditions": "Sunny", "Road_traffic_density": "Low", "Vehicle_condition": 1,
        "Type_of_order": "Meal", "Type_of_vehicle": "motorcycle", "Festival": "No", "City": "Urban",
        "Time_Orderd": "12:00"}] * 5)
    config = tmp_path / "preprocessing.json"
    Preprocessor().fit(raw).save(config)
    context = SimpleNamespace(artifacts={"checkpoint": checkpoint, "fitted_state": str(state),
        "preprocessing": str(config)}, model_config={"batch_size": 2})
    wrapped = RawTabPFNModel()
    wrapped.load_context(context)
    result = wrapped.predict(context, raw)
    assert loaded_params["model_path"] == checkpoint
    assert calls == [2, 2, 1]
    np.testing.assert_array_equal(result, np.full(5, 27.0))
    with zipfile.ZipFile(state) as archive:
        assert json.loads(archive.read("init_params.json"))["model_path"] == "old-machine/checkpoint.safetensors"
