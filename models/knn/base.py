import numpy as np
from math_utils.statistics import distance


class BaseKNNModel:
    def __init__(self, k: int = 5):
        if k <= 0:
            raise ValueError("k must be a positive integer.")

        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array.")
        if len(X) != len(y):
            raise ValueError(f"X and y have mismatched lengths: {len(X)} vs {len(y)}")
        if len(X) == 0:
            raise ValueError("Training data must not be empty.")
        if np.isnan(X).any():
            raise ValueError("KNNModel does not accept NaN values in X.")

        self.X_train = X
        self.y_train = y
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.X_train is None or self.y_train is None:
            raise ValueError("Model not fitted yet.")

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if X.shape[1] != self.X_train.shape[1]:
            raise ValueError(
                f"Expected {self.X_train.shape[1]} features, got {X.shape[1]}."
            )
        if np.isnan(X).any():
            raise ValueError("KNNModel does not accept NaN values in X.")

        return np.array([self._predict(x) for x in X])

    def _predict(self, x: np.ndarray):
        distances = np.array([distance(x, x_train) for x_train in self.X_train])
        k_indices = np.argsort(distances)[: min(self.k, len(self.X_train))]
        k_nearest_labels = self.y_train[k_indices]
        return self._decide(k_nearest_labels)

    def _decide(self, k_nearest_labels: np.ndarray):
        raise NotImplementedError
