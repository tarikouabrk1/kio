import numpy as np
from .decisiontree import DecisionTree


class RandomForest:
    """
    Ensemble of Decision Trees trained via Bootstrap Aggregation (Bagging).

    Key features
    ------------
    - Feature subspace sampling: each split considers only √(n_features)
      randomly chosen features (classification) or n_features/3 (regression).
    - Bootstrap sampling: each tree sees a random sample with replacement.
    - Aggregation: majority vote (classification) or mean (regression).

    Parameters
    ----------
    n_estimators      : int   — number of trees
    max_depth         : int | None
    regression        : bool
    min_samples_split : int
    max_features      : str | int | None
        'sqrt'  — √n_features  (default for classification)
        'third' — n_features // 3  (default for regression)
        int     — exact number of features
        None    — use all features (disables subspace sampling)
    seed              : int | None — random seed for reproducibility
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth=None,
        regression: bool = False,
        min_samples_split: int = 2,
        max_features="sqrt",
        seed: int = None,
    ):
        self.n_estimators      = n_estimators
        self.max_depth         = max_depth
        self.regression        = regression
        self.min_samples_split = min_samples_split
        self.max_features      = max_features
        self.seed              = seed
        self.trees             = []
        self._feature_indices  = []   # list of feature index arrays per tree

    def _n_features_to_use(self, n_features: int) -> int:
        if self.max_features is None:
            return n_features
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        if self.max_features == "third":
            return max(1, n_features // 3)
        if isinstance(self.max_features, int):
            return max(1, min(self.max_features, n_features))
        raise ValueError(f"Unknown max_features: {self.max_features}")

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        rng = np.random.default_rng(self.seed)
        n_samples, n_features = X.shape
        k = self._n_features_to_use(n_features)

        self.trees            = []
        self._feature_indices = []

        for _ in range(self.n_estimators):
            # Bootstrap sample
            row_idx  = rng.integers(0, n_samples, size=n_samples)
            # Feature subspace
            feat_idx = rng.choice(n_features, size=k, replace=False)

            X_sample = X[np.ix_(row_idx, feat_idx)]
            y_sample = y[row_idx]

            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                regression=self.regression,
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
            self._feature_indices.append(feat_idx)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.trees:
            raise ValueError("RandomForest has not been fitted yet.")
        X = np.asarray(X, dtype=np.float64)

        # Collect predictions from all trees (n_estimators × n_samples)
        all_preds = np.array([
            tree.predict(X[:, feat_idx])
            for tree, feat_idx in zip(self.trees, self._feature_indices)
        ])

        if self.regression:
            return np.mean(all_preds, axis=0)

        # Classification: majority vote per sample using NumPy
        n_samples = X.shape[0]
        result    = np.empty(n_samples, dtype=all_preds.dtype)
        for i in range(n_samples):
            votes      = all_preds[:, i]
            vals, cnts = np.unique(votes, return_counts=True)
            result[i]  = vals[np.argmax(cnts)]
        return result

    def feature_importance(self, n_features_total: int) -> np.ndarray:
        """
        Aggregate feature importances across all trees.

        Parameters
        ----------
        n_features_total : int — original number of features in X

        Returns
        -------
        importance : np.ndarray, shape (n_features_total,), sum = 1
        """
        if not self.trees:
            raise ValueError("RandomForest has not been fitted yet.")
        importance = np.zeros(n_features_total)
        for tree, feat_idx in zip(self.trees, self._feature_indices):
            local_imp = tree.feature_importance()
            for local_i, global_i in enumerate(feat_idx):
                if local_i < len(local_imp):
                    importance[global_i] += local_imp[local_i]
        total = importance.sum()
        return importance / total if total > 0 else importance