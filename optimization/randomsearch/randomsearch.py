import numpy as np
from ..base import BaseOptimizer
from ..kfold.kfold import KFold


class RandomSearch(BaseOptimizer):

    def __init__(self, model_class, param_distributions, metric,n_iter=10, greater_is_better=True, cv: int = None,shuffle=False, seed=None):
        self.model_class = model_class
        self.param_distributions = param_distributions
        self.metric = metric
        self.n_iter = n_iter
        self.greater_is_better = greater_is_better
        self.cv = cv
        self.kfold = KFold(n_splits=cv, shuffle=shuffle, seed=seed) if cv else None
        self.rng = np.random.default_rng(seed)

        self.best_score = -np.inf if greater_is_better else np.inf
        self.best_params = None
        self.best_model = None
        self.results = []

    def _is_better(self, score):
        if self.greater_is_better:
            return score > self.best_score
        return score < self.best_score

    def _sample_params(self):
        return {
            key: self.rng.choice(values)
            for key, values in self.param_distributions.items()
        }

    def _score_combo(self, params, X_train, y_train, X_val, y_val):
        """Evaluate a param combo — with CV or a fixed val split."""
        if self.kfold:
            X_all = np.concatenate([X_train, X_val])
            y_all = np.concatenate([y_train, y_val])
            fold_scores = []
            for X_tr, y_tr, X_v, y_v in self.kfold.split_data(X_all, y_all):
                model = self.model_class(**params)
                model.fit(X_tr, y_tr)
                fold_scores.append(self._evaluate(model, X_v, y_v, self.metric))
            return float(np.mean(fold_scores)), float(np.std(fold_scores)), fold_scores
        else:
            model = self.model_class(**params)
            model.fit(X_train, y_train)
            score = self._evaluate(model, X_val, y_val, self.metric)
            return score, None, None

    def search(self, X_train, y_train, X_val=None, y_val=None):
        if not self.kfold and (X_val is None or y_val is None):
            raise ValueError("Provide X_val/y_val when cv is not set.")

        for _ in range(self.n_iter):
            params = self._sample_params()
            score, std, fold_scores = self._score_combo(params, X_train, y_train, X_val, y_val)

            self.results.append({
                "params": params,
                "score": score,
                "std": std,
                "fold_scores": fold_scores,
            })

            if self._is_better(score):
                self.best_score = score
                self.best_params = params
                self.best_model = self.model_class(**params)
                self.best_model.fit(X_train, y_train)

        return self.best_model, self.best_params, self.best_score