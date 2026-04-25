from abc import ABC, abstractmethod
import numpy as np


class ImputerStrategy(ABC):
    @abstractmethod
    def compute_value(self, values: np.ndarray):
        pass


def _missing_mask(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)

    if np.issubdtype(values.dtype, np.number):
        return np.isnan(values)

    return np.array(
        [value is None or (isinstance(value, float) and np.isnan(value)) for value in values],
        dtype=bool,
    )


def _non_missing(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    return values[~_missing_mask(values)]


class MeanImputer(ImputerStrategy):
    def compute_value(self, values: np.ndarray) -> np.float64:
        clean_values = _non_missing(np.asarray(values, dtype=np.float64))
        if clean_values.size == 0:
            raise ValueError("Cannot compute mean for an all-missing column.")
        return np.mean(clean_values)

class MedianImputer(ImputerStrategy):
    def compute_value(self, values: np.ndarray) -> np.float64:
        clean_values = _non_missing(np.asarray(values, dtype=np.float64))
        if clean_values.size == 0:
            raise ValueError("Cannot compute median for an all-missing column.")
        return np.median(clean_values)

class ModeImputer(ImputerStrategy):
    def compute_value(self, values: np.ndarray):
        clean_values = _non_missing(values)
        if clean_values.size == 0:
            raise ValueError("Cannot compute mode for an all-missing column.")

        unique_values, counts = np.unique(clean_values, return_counts=True)
        return unique_values[np.argmax(counts)]

class ConstantImputer(ImputerStrategy):
    def __init__(self, constant_value):
        self.constant_value = constant_value
        
    def compute_value(self, values):
        return self.constant_value


# Simple Imputer
class SimpleImputer:
    def __init__(self, strategy: ImputerStrategy):
        self.strategy = strategy
        self.value = None
        self._is_fitted = False
        
    def fit(self, X: np.ndarray):
        X = np.asarray(X)

        if X.ndim == 1:
            self.value = self.strategy.compute_value(X)
        elif X.ndim == 2:
            self.value = [self.strategy.compute_value(X[:, col]) for col in range(X.shape[1])]
        else:
            raise ValueError("SimpleImputer expects a 1D or 2D array.")

        self._is_fitted = True
        return self
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise ValueError("The imputer has not been fitted yet.")

        X = np.asarray(X).copy()

        if X.ndim == 1:
            mask = _missing_mask(X)
            X[mask] = self.value
            return X

        if X.ndim == 2:
            if len(self.value) != X.shape[1]:
                raise ValueError(
                    f"Expected {len(self.value)} columns, got {X.shape[1]}."
                )

            for col, fill_value in enumerate(self.value):
                mask = _missing_mask(X[:, col])
                X[mask, col] = fill_value
            return X

        raise ValueError("SimpleImputer expects a 1D or 2D array.")

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)
    
