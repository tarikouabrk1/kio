import numpy as np
from .base import BaseEncoder

# =========================================================
# Label Encoder
# =========================================================
class LabelEncoder(BaseEncoder):
    """
    Encodes categorical values into integers.
    Supports optional ordering and unknown handling.
    """

    def __init__(self, order: list | None = None, handle_unknown: str = "error"):
        if handle_unknown not in ("error", "ignore"):
            raise ValueError("handle_unknown must be 'error' or 'ignore'.")

        self.order = order
        self.handle_unknown = handle_unknown

        self.classes_ = None
        self._label_to_int = None
        self._int_to_label = None

    def fit(self, X: np.ndarray):
        X = np.asarray(X).ravel()

        if self.order is not None:
            missing = set(X) - set(self.order)
            if missing:
                raise ValueError(f"Missing values in order: {missing}")
            self.classes_ = list(self.order)
        else:
            self.classes_ = list(np.unique(X))

        self._label_to_int = {
            label: i for i, label in enumerate(self.classes_)
        }

        self._int_to_label = {
            i: label for label, i in self._label_to_int.items()
        }

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise ValueError("LabelEncoder not fitted.")

        X = np.asarray(X).ravel()
        result = np.empty(len(X), dtype=np.int64)

        for i, value in enumerate(X):
            code = self._label_to_int.get(value)

            if code is None:
                if self.handle_unknown == "error":
                    raise ValueError(f"Unknown category: {value}")
                result[i] = -1
            else:
                result[i] = code

        return result

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        if self._int_to_label is None:
            raise ValueError("LabelEncoder not fitted.")

        X = np.asarray(X).ravel()
        return np.array(
            [self._int_to_label.get(int(code), None) for code in X],
            dtype=object
        )
