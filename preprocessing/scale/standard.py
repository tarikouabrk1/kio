import numpy as np


class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None
        
    def fit(self, X: np.array):
        self.mean = np.mean(X)
        self.std = np.std(X, ddof=1) # 1 for Bessel's correction regarding standard deviation
        
    def transform(self, X: np.array) -> np.array:
        if self.mean is None or self.std is None:
            raise ValueError("The scaler has not been fitted yet.")
        return (X - self.mean) / self.std
    
    def fit_transform(self, X: np.array) -> np.array:
        self.fit(X)
        return self.transform(X)