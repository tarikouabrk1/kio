import numpy as np


class MinMaxScaler:
    def __init__(self):
        self.min = None
        self.max = None

    def fit(self, X: np.array):
        self.min = np.min(X)
        self.max = np.max(X)

    def transform(self, X: np.array) -> np.array:
        if self.min is None or self.max is None:
            raise ValueError("The scaler has not been fitted yet.")
        return (X - self.min) / (self.max - self.min)

    def fit_transform(self, X: np.array) -> np.array:
        self.fit(X)
        return self.transform(X)
