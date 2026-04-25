import numpy as np

from .base import BaseKNNModel


class KNNClassifier(BaseKNNModel):
    def _decide(self, k_nearest_labels: np.ndarray):
        unique_labels, counts = np.unique(k_nearest_labels, return_counts=True)
        return unique_labels[np.argmax(counts)]
