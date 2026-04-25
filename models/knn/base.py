import numpy as np
from math_utils.statistics import distance


class BaseKNNModel:
    """
    Base class for KNN classifier and regressor.

    Parameters
    ----------
    k      : int — number of nearest neighbours
    metric : str — 'euclidean' (default) | 'manhattan'
    """

    def __init__(self, k: int = 5, metric: str = "euclidean"):
        if k <= 0:
            raise ValueError("k must be a positive integer.")
        if metric not in ("euclidean", "manhattan"):
            raise ValueError("metric must be 'euclidean' or 'manhattan'.")
        self.k       = k
        self.metric  = metric
        self.X_train = None
        self.y_train = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must be a 2-D array.")
        if y.ndim != 1:
            raise ValueError("y must be a 1-D array.")
        if len(X) != len(y):
            raise ValueError(f"X and y length mismatch: {len(X)} vs {len(y)}")
        if len(X) == 0:
            raise ValueError("Training data must not be empty.")
        if np.isnan(X).any():
            raise ValueError("KNN does not accept NaN values in X.")

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
            raise ValueError("X must be a 2-D array.")
        if X.shape[1] != self.X_train.shape[1]:
            raise ValueError(
                f"Expected {self.X_train.shape[1]} features, got {X.shape[1]}."
            )
        if np.isnan(X).any():
            raise ValueError("KNN does not accept NaN values in X.")

        return np.array([self._predict_one(x) for x in X])

    def _predict_one(self, x: np.ndarray):
        distances  = np.array([
            distance(x, x_train, metric=self.metric)
            for x_train in self.X_train
        ])
        k_indices        = np.argsort(distances)[: min(self.k, len(self.X_train))]
        k_nearest_labels = self.y_train[k_indices]
        return self._decide(k_nearest_labels)

    def _decide(self, k_nearest_labels: np.ndarray):
        raise NotImplementedError