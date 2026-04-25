import numpy as np
from .base import BaseEncoder

# =========================================================
# One Hot Encoder
# =========================================================
class OneHotEncoder(BaseEncoder):
    """
    Encodes categorical features into binary vectors.
    """

    def __init__(self, handle_unknown: str = "error"):
        if handle_unknown not in ("error", "ignore"):
            raise ValueError("handle_unknown must be 'error' or 'ignore'.")

        self.handle_unknown = handle_unknown
        self.categories_ = None
        self._category_to_index = None

    def fit(self, X: np.ndarray):
        X = np.asarray(X).ravel()

        self.categories_ = np.unique(X)
        self._category_to_index = {
            cat: i for i, cat in enumerate(self.categories_)
        }
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.categories_ is None:
            raise ValueError("OneHotEncoder has not been fitted yet.")

        X = np.asarray(X).ravel()

        n_samples = X.shape[0]
        n_cats = len(self.categories_)

        result = np.zeros((n_samples, n_cats), dtype=np.float64)

        for i, value in enumerate(X):
            idx = self._category_to_index.get(value)

            if idx is None:
                if self.handle_unknown == "error":
                    raise ValueError(f"Unknown category: {value}")
                continue

            result[i, idx] = 1.0

        return result

    def get_feature_names(self, prefix: str = "x"):
        if self.categories_ is None:
            raise ValueError("Encoder not fitted.")

        return [f"{prefix}_{cat}" for cat in self.categories_]
