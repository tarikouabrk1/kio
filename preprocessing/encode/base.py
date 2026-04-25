from abc import ABC, abstractmethod
import numpy as np


# =========================================================
# Base Encoder (Interface)
# =========================================================
class BaseEncoder(ABC):

    @abstractmethod
    def fit(self, X: np.ndarray):
        pass

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        pass

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)
