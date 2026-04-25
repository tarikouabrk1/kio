from decisionstumps import DecisionStump, gini_impurity
import numpy as np


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, regression=False):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.regression = regression

    def _leaf_value(self, y):
        if self.regression:
            return y.mean()
        return np.bincount(y).argmax()

    def _best_split(self, X, y):
        stump = DecisionStump()
        stump.fit(X, y)
        return stump.feature, stump.threshold

    def _build(self, X, y, depth):
        if (
            len(y) < self.min_samples_split or
            (self.max_depth is not None and depth >= self.max_depth) or
            len(np.unique(y)) == 1
        ):
            return Node(value=self._leaf_value(y))

        feature, threshold = self._best_split(X, y)

        if feature is None:
            return Node(value=self._leaf_value(y))

        left_mask = X[:, feature] <= threshold
        left = self._build(X[left_mask], y[left_mask], depth + 1)
        right = self._build(X[~left_mask], y[~left_mask], depth + 1)

        return Node(feature=feature, threshold=threshold, left=left, right=right)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.regression = np.issubdtype(y.dtype, np.floating)
        self.root = self._build(X, y, depth=0)

    def _predict_one(self, x, node):
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.root is None:
            raise ValueError("Tree has not been fitted yet.")
        return np.array([self._predict_one(x, self.root) for x in X])

