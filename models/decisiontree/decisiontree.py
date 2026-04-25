import numpy as np
from .decisionstumps import DecisionStump, gini_impurity, mse_variance


class Node:
    def __init__(self, feature=None, threshold=None,
                 left=None, right=None, value=None):
        self.feature   = feature
        self.threshold = threshold
        self.left      = left
        self.right     = right
        self.value     = value

    def is_leaf(self):
        return self.value is not None


class DecisionTree:
    """
    Full recursive decision tree supporting classification and regression.

    Parameters
    ----------
    max_depth         : int | None
    min_samples_split : int
    regression        : bool   — if True, uses MSE criterion; else Gini
    """

    def __init__(self, max_depth=None, min_samples_split=2, regression=False):
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.regression        = regression
        self.root              = None
        self._criterion        = "mse" if regression else "gini"

    # ---- helpers ------------------------------------------------

    def _leaf_value(self, y):
        if self.regression:
            return float(y.mean())
        return int(np.bincount(y.astype(int)).argmax())

    def _best_split(self, X, y):
        stump = DecisionStump(criterion=self._criterion)
        stump.fit(X, y)
        return stump.feature, stump.threshold

    def _is_pure(self, y):
        if self.regression:
            return float(np.std(y)) < 1e-10
        return len(np.unique(y)) == 1

    # ---- recursive builder --------------------------------------

    def _build(self, X, y, depth):
        if (
            len(y) < self.min_samples_split
            or (self.max_depth is not None and depth >= self.max_depth)
            or self._is_pure(y)
        ):
            return Node(value=self._leaf_value(y))

        feature, threshold = self._best_split(X, y)

        if feature is None:
            return Node(value=self._leaf_value(y))

        left_mask  = X[:, feature] <= threshold
        right_mask = ~left_mask

        # Guard: avoid degenerate splits (can happen with constant columns)
        if left_mask.sum() == 0 or right_mask.sum() == 0:
            return Node(value=self._leaf_value(y))

        left  = self._build(X[left_mask],  y[left_mask],  depth + 1)
        right = self._build(X[right_mask], y[right_mask], depth + 1)

        return Node(feature=feature, threshold=threshold,
                    left=left, right=right)

    # ---- public API ---------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        # Auto-detect regression from dtype if not explicitly set
        if np.issubdtype(y.dtype, np.floating):
            self.regression = True
            self._criterion = "mse"
        self.root = self._build(X, y, depth=0)
        return self

    def _predict_one(self, x, node):
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.root is None:
            raise ValueError("Tree has not been fitted yet.")
        X = np.asarray(X, dtype=np.float64)
        return np.array([self._predict_one(x, self.root) for x in X])

    # ---- feature importance -------------------------------------

    def feature_importance(self) -> np.ndarray:
        """
        Returns a 1-D array of feature importance scores (sum = 1).
        Uses impurity decrease, accumulated during tree traversal.
        """
        if self.root is None:
            raise ValueError("Tree has not been fitted yet.")
        scores = {}
        self._importance_traverse(self.root, scores)
        if not scores:
            return np.array([])
        max_feat = max(scores.keys())
        importance = np.zeros(max_feat + 1)
        for f, v in scores.items():
            importance[f] = v
        total = importance.sum()
        return importance / total if total > 0 else importance

    def _importance_traverse(self, node, scores):
        if node.is_leaf():
            return
        f = node.feature
        scores[f] = scores.get(f, 0.0) + 1.0   # count splits per feature
        self._importance_traverse(node.left,  scores)
        self._importance_traverse(node.right, scores)