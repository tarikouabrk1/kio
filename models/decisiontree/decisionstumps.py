import numpy as np


def gini_impurity(y: np.ndarray) -> float:
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return 1 - (probs**2).sum()


class DecisionStump:
    def __init__(self):
        self.feature = None
        self.threshold = None
        self.left_value = None
        self.right_value = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        best_gini = np.inf
        n_samples, n_features = X.shape

        for f in range(n_features):
            col = X[:, f]
            thresholds = np.unique(col)

            for t in thresholds:
                left_mask = col <= t
                right_mask = ~left_mask

                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue

                gini = (
                    gini_impurity(y[left_mask]) * left_mask.sum()
                    + gini_impurity(y[right_mask]) * right_mask.sum()
                ) / n_samples

                if gini < best_gini:
                    best_gini = gini
                    self.feature = f
                    self.threshold = t
                    self.left_value = np.bincount(y[left_mask]).argmax()
                    self.right_value = np.bincount(y[right_mask]).argmax()

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.feature is None:
            raise ValueError("Stump has not been fitted yet.")
        col = X[:, self.feature]
        return np.where(col <= self.threshold, self.left_value, self.right_value)
