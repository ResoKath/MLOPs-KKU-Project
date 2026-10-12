"""A deployable artifact includes preprocessing, not just estimator weights."""
import numpy as np


class DeliveryModel:
    def __init__(self, preprocessor, estimator, metadata=None):
        self.preprocessor = preprocessor
        self.estimator = estimator
        self.metadata = metadata or {}

    def predict(self, raw):
        result = np.asarray(self.estimator.predict(self.preprocessor.transform(raw)), dtype=float)
        if not np.isfinite(result).all() or (result <= 0).any():
            raise ValueError("Model produced invalid delivery time")
        return result
