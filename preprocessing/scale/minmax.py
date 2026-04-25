import numpy as np


class MinMaxScaler:
    """
    Scales features to the [0, 1] range.

    Supports both 1-D arrays (single feature) and 2-D arrays
    (multiple features — per-column min / max).
    """

    def __init__(self):
        self.min_ = None
        self.max_ = None

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=np.float64)

        if X.ndim == 1:
            self.min_ = np.min(X)
            self.max_ = np.max(X)
        elif X.ndim == 2:
            self.min_ = np.min(X, axis=0)
            self.max_ = np.max(X, axis=0)
        else:
            raise ValueError("MinMaxScaler expects a 1-D or 2-D array.")

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.min_ is None or self.max_ is None:
            raise ValueError("MinMaxScaler has not been fitted yet.")
        X = np.asarray(X, dtype=np.float64)
        denom = self.max_ - self.min_
        # Constant columns stay at 0
        if np.ndim(denom) == 0:
            denom = denom if denom > 1e-8 else 1.0
        else:
            denom = np.where(denom < 1e-8, 1.0, denom)
        return (X - self.min_) / denom

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        if self.min_ is None or self.max_ is None:
            raise ValueError("MinMaxScaler has not been fitted yet.")
        X = np.asarray(X, dtype=np.float64)
        return X * (self.max_ - self.min_) + self.min_