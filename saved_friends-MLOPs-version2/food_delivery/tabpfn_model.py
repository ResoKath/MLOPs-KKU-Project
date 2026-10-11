"""Portable raw-input MLflow wrapper for native fitted TabPFN state."""
import json
import os
import tempfile
import zipfile
from pathlib import Path

import mlflow.pyfunc
import numpy as np

from food_delivery.preprocessing import Preprocessor


class RawTabPFNModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        from tabpfn.model_loading import load_fitted_tabpfn_model

        self.processor = Preprocessor.load(context.artifacts["preprocessing"])
        self.batch_size = int(context.model_config.get("batch_size", 16))
        # Native fitted archives contain an initialization checkpoint path.
        # Rewrite only that path to the checkpoint packaged in this MLflow model.
        with tempfile.TemporaryDirectory() as temporary:
            relocated = Path(temporary) / "portable.tabpfn_fit"
            with zipfile.ZipFile(context.artifacts["fitted_state"]) as source, zipfile.ZipFile(relocated, "w") as target:
                for name in source.namelist():
                    value = source.read(name)
                    if name == "init_params.json":
                        params = json.loads(value)
                        params["model_path"] = context.artifacts["checkpoint"]
                        value = json.dumps(params).encode("utf8")
                    target.writestr(name, value)
            self.estimator = load_fitted_tabpfn_model(relocated, device=os.environ.get("TABPFN_SERVING_DEVICE", "cuda"))

    def predict(self, context, model_input, params=None):
        features = self.processor.transform(model_input)
        output = np.concatenate([self.estimator.predict(features.iloc[i:i + self.batch_size])
                                 for i in range(0, len(features), self.batch_size)])
        if not np.isfinite(output).all() or (output <= 0).any():
            raise ValueError("TabPFN returned invalid delivery time")
        return output
