import numpy as np
from ..base import BaseOptimizer

class KFold:
    """
    K-Fold Cross Validation splitter.
    Splits data into K folds and yields (train, val) index pairs.
    """
    def __init__(self, n_splits: int = 5, shuffle: bool = False, seed: int = None):
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2.")

        self.n_splits = n_splits
        self.shuffle = shuffle
        self.seed = seed

    def split(self, X: np.ndarray):
        """
        Yields (train_indices, val_indices) for each fold.

        Usage:
            for X_train, y_train, X_val, y_val in kf.split_data(X, y):
                ...
        """
        n_samples = len(X)

        if n_samples < self.n_splits:
            raise ValueError(
                f"Not enough samples ({n_samples}) for {self.n_splits} splits."
            )

        indices = np.arange(n_samples)

        if self.shuffle:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(indices)

        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits)
        fold_sizes[: n_samples % self.n_splits] += 1  # distribute remainder

        current = 0
        for fold_size in fold_sizes:
            val_indices = indices[current : current + fold_size]
            train_indices = np.concatenate(
                [indices[:current], indices[current + fold_size :]]
            )
            yield train_indices, val_indices
            current += fold_size

    def split_data(self, X: np.ndarray, y: np.ndarray):
        """
        Convenience wrapper — yields (X_train, y_train, X_val, y_val) directly.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        for train_idx, val_idx in self.split(X):
            yield X[train_idx], y[train_idx], X[val_idx], y[val_idx]


class KFoldSearch(BaseOptimizer):
    """
    Wraps GridSearch or RandomSearch with K-Fold cross-validation.
    Evaluates each param combo across K folds and returns the best average score.
    """

    def __init__(self, model_class, param_grid, metric, n_splits=5, shuffle=False, seed=None):
        self.model_class = model_class
        self.param_grid = param_grid
        self.metric = metric
        self.kfold = KFold(n_splits=n_splits, shuffle=shuffle, seed=seed)

        self.best_score = -np.inf
        self.best_params = None
        self.best_model = None
        self.results = []

    def _get_param_combinations(self):
        raise NotImplementedError

    def search(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        for params in self._get_param_combinations():
            fold_scores = []

            for X_train, y_train, X_val, y_val in self.kfold.split_data(X, y):
                model = self.model_class(**params)
                model.fit(X_train, y_train)
                score = self._evaluate(model, X_val, y_val, self.metric)
                fold_scores.append(score)

            avg_score = float(np.mean(fold_scores))
            std_score = float(np.std(fold_scores))

            self.results.append({
                "params": params,
                "mean_score": avg_score,
                "std_score": std_score,
                "fold_scores": fold_scores,
            })

            if avg_score > self.best_score:
                self.best_score = avg_score
                self.best_params = params
                # Refit on full data with best params
                self.best_model = self.model_class(**params)
                self.best_model.fit(X, y)

        return self.best_model, self.best_params, self.best_score


class KFoldGridSearch(KFoldSearch):
    """Grid Search with K-Fold CV."""

    def _get_param_combinations(self):
        import itertools
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        for combo in itertools.product(*values):
            yield dict(zip(keys, combo))


class KFoldRandomSearch(KFoldSearch):
    """Random Search with K-Fold CV."""

    def __init__(self, model_class, param_distributions, metric,n_iter=10, n_splits=5, shuffle=False, seed=None):
        super().__init__(model_class, param_distributions, metric, n_splits, shuffle, seed)
        self.n_iter = n_iter
        self.rng = np.random.default_rng(seed)

    def _get_param_combinations(self):
        for _ in range(self.n_iter):
            yield {
                key: self.rng.choice(values)
                for key, values in self.param_grid.items()
            }