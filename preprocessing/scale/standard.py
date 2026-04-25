import numpy as np


class StandardScaler:
    """
    Standardises features to zero mean and unit variance.

    Supports both 1-D arrays (single feature) and 2-D arrays
    (multiple features — per-column mean / std).

    Parameters
    ----------
    ddof : int
        Delta degrees of freedom used for std calculation.
        Default 1 (sample std, matching sklearn).
    """

    def __init__(self, ddof: int = 1):
        self.ddof = ddof
        self.mean_ = None
        self.std_ = None
        self._ndim = None

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        self._ndim = X.ndim

        if X.ndim == 1:
            self.mean_ = np.mean(X)
            self.std_ = np.std(X, ddof=self.ddof)
            if self.std_ < 1e-8:
                self.std_ = 1.0          # constant column — avoid div-by-zero
        elif X.ndim == 2:
            self.mean_ = np.mean(X, axis=0)
            self.std_ = np.std(X, axis=0, ddof=self.ddof)
            self.std_[self.std_ < 1e-8] = 1.0
        else:
            raise ValueError("StandardScaler expects a 1-D or 2-D array.")

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise ValueError("StandardScaler has not been fitted yet.")
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise ValueError("StandardScaler has not been fitted yet.")
        X = np.asarray(X, dtype=np.float64)
        return X * self.std_ + self.mean_