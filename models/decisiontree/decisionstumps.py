import numpy as np


# =========================================================
# Impurity / variance helpers
# =========================================================

def gini_impurity(y: np.ndarray) -> float:
    if len(y) == 0:
        return 0.0
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return float(1 - (probs ** 2).sum())


def mse_variance(y: np.ndarray) -> float:
    """Mean squared deviation from the column mean — used for regression splits."""
    if len(y) == 0:
        return 0.0
    return float(np.mean((y - y.mean()) ** 2))


# =========================================================
# Decision Stump
# =========================================================

class DecisionStump:
    """
    Single-level decision tree (depth-1) used as the split-finder
    inside the full DecisionTree.

    Parameters
    ----------
    criterion : str
        'gini'  — Gini impurity (classification)
        'mse'   — Mean squared error variance reduction (regression)
    """

    def __init__(self, criterion: str = "gini"):
        if criterion not in ("gini", "mse"):
            raise ValueError("criterion must be 'gini' or 'mse'.")
        self.criterion = criterion
        self.feature = None
        self.threshold = None
        self.left_value = None
        self.right_value = None

    def _impurity(self, y: np.ndarray) -> float:
        if self.criterion == "gini":
            return gini_impurity(y)
        return mse_variance(y)

    def _leaf_value(self, y: np.ndarray):
        if self.criterion == "gini":
            return int(np.bincount(y.astype(int)).argmax())
        return float(y.mean())

    def fit(self, X: np.ndarray, y: np.ndarray):
        best_score = np.inf
        n_samples, n_features = X.shape

        for f in range(n_features):
            col = X[:, f]
            thresholds = np.unique(col)

            for t in thresholds:
                left_mask  = col <= t
                right_mask = ~left_mask

                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue

                score = (
                    self._impurity(y[left_mask])  * left_mask.sum()
                    + self._impurity(y[right_mask]) * right_mask.sum()
                ) / n_samples

                if score < best_score:
                    best_score      = score
                    self.feature    = f
                    self.threshold  = t
                    self.left_value  = self._leaf_value(y[left_mask])
                    self.right_value = self._leaf_value(y[right_mask])

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.feature is None:
            raise ValueError("Stump has not been fitted yet.")
        col = X[:, self.feature]
        return np.where(col <= self.threshold, self.left_value, self.right_value)