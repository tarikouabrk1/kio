import numpy as np

from .base import BaseKNNModel


class KNNRegressor(BaseKNNModel):
    def _decide(self, k_nearest_labels: np.ndarray):
        return float(np.mean(k_nearest_labels.astype(float)))
