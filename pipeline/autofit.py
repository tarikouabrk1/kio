"""
AutoFit Pipeline
================
Orchestrates the full GlassBox workflow:

    EDA → Imputation → Encoding → Scaling → Model Search → Report

Usage
-----
    from pipeline.autofit import AutoFit

    result = AutoFit(target="churned", task="classification").fit(df)
    print(result)                 # JSON-serialisable dict
    result["best_model"].predict(X_new)
"""

import numpy as np
import time

from eda.inspector import EDAInspector
from preprocessing.impute.simple import SimpleImputer, MedianImputer, ModeImputer
from preprocessing.encode.label import LabelEncoder
from preprocessing.encode.onehot import OneHotEncoder
from preprocessing.scale.standard import StandardScaler
from models.logisticregression.logistic_regression import LogisticRegression
from models.linearregression.linear import LinearRegression
from models.knn.classifier import KNNClassifier
from models.knn.regressor import KNNRegressor
from models.naive.naive import NaiveBayes
from models.decisiontree.decisiontreeclassifier import DecisionTreeClassifier
from models.decisiontree.decisiontreeregression import DecisionTreeRegression
from models.decisiontree.randomforestclassifier import RandomForestClassifier
from models.decisiontree.randomforestregression import RandomForestRegression
from optimization.gridsearch.gridsearch import GridSearch
from optimization.randomsearch.randomsearch import RandomSearch
from evaluation.classification.classification import (
    accuracy, classification_report, confusion_matrix,
)
from evaluation.regression.regression import regression_report


# =========================================================
# Candidate model grids
# =========================================================

_CLASSIFICATION_CANDIDATES = [
    (LogisticRegression,      {"lr": [0.05, 0.1], "max_iters": [500, 1000]}),
    (KNNClassifier,           {"k": [3, 5, 7]}),
    (DecisionTreeClassifier,  {"max_depth": [3, 5, None]}),
    (RandomForestClassifier,  {"n_estimators": [50, 100], "max_depth": [5, None]}),
    (NaiveBayes,              {}),
]

_REGRESSION_CANDIDATES = [
    (LinearRegression,       {"lr": [0.01, 0.05], "epochs": [1000, 3000]}),
    (KNNRegressor,           {"k": [3, 5]}),
    (DecisionTreeRegression, {"max_depth": [3, 5, None]}),
    (RandomForestRegression, {"n_estimators": [50, 100], "max_depth": [5, None]}),
]


# =========================================================
# AutoFit
# =========================================================

class AutoFit:
    """
    End-to-end automated ML pipeline.

    Parameters
    ----------
    target    : str  — name of the target column in the DataFrame
    task      : str  — 'classification' | 'regression' | 'auto'
                       'auto' infers from the target column dtype
    cv        : int  — number of KFold splits (default 3)
    test_size : float — fraction of data held out for final evaluation
    seed      : int | None
    verbose   : bool
    """

    def __init__(
        self,
        target: str,
        task: str = "auto",
        cv: int = 3,
        test_size: float = 0.2,
        seed: int = 42,
        search_strategy: str = "grid",
        random_iter: int = 10,
        verbose: bool = True,
    ):
        if task not in ("auto", "classification", "regression"):
            raise ValueError("task must be 'auto', 'classification', or 'regression'.")
        if search_strategy not in ("grid", "random"):
            raise ValueError("search_strategy must be 'grid' or 'random'.")
        self.target    = target
        self.task      = task
        self.cv        = cv
        self.test_size = test_size
        self.seed      = seed
        self.search_strategy = search_strategy
        self.random_iter = random_iter
        self.verbose   = verbose

        # Artifacts populated during fit
        self._eda_report      = None
        self._feature_names   = []
        self._encoders        = {}
        self._imputers        = {}
        self._scaler          = None
        self.best_model       = None
        self.best_params      = None
        self.best_model_name  = None
        self._report          = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, df) -> dict:
        """
        Run the full pipeline on a GlassBox DataFrame.

        Returns a JSON-serialisable report dict. The fitted best model
        is also stored in self.best_model.
        """
        t_start = time.time()

        # 1. EDA
        if self.verbose:
            print("[AutoFit] Running EDA inspection…")
        inspector = EDAInspector(df)
        self._eda_report = inspector.run()

        # 2. Resolve task
        task = self._resolve_task(df)
        if self.verbose:
            print(f"[AutoFit] Task detected: {task}")

        # 3. Build feature matrix
        if self.verbose:
            print("[AutoFit] Building feature matrix…")
        X, y, feature_names = self._build_features(df, task)
        self._feature_names = feature_names

        # 4. Train / test split (stratified by index for simplicity)
        X_train, X_test, y_train, y_test = self._split(X, y)

        # 5. Model search
        if self.verbose:
            print("[AutoFit] Searching models…")
        search_results = self._search(X_train, y_train, X_test, y_test, task)

        # 6. Evaluate best model on hold-out
        y_pred = self.best_model.predict(X_test)
        if task == "classification":
            eval_metrics = classification_report(y_test, y_pred)
            eval_metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()
        else:
            eval_metrics = regression_report(y_test, y_pred)

        # 7. Feature importance
        feat_importance = self._get_feature_importance(X_train)

        elapsed = round(time.time() - t_start, 2)
        if self.verbose:
            print(f"[AutoFit] Done in {elapsed}s. Best model: {self.best_model_name}")

        self._report = {
            "task":              task,
            "target":            self.target,
            "n_samples":         int(X.shape[0]),
            "n_features":        int(X.shape[1]),
            "feature_names":     feature_names,
            "best_model":        self.best_model_name,
            "best_params":       self.best_params,
            "eval_metrics":      eval_metrics,
            "feature_importance": feat_importance,
            "search_results":    search_results,
            "eda_summary": {
                "shape":           self._eda_report["shape"],
                "missing_columns": [
                    col for col, info in self._eda_report["missing_summary"].items()
                    if info["missing_count"] > 0
                ],
                "top_correlations": self._eda_report["correlations"].get("top_pairs", [])[:5],
                "outlier_columns": [
                    col for col, info in self._eda_report["outliers"].items()
                    if info["n_outliers"] > 0
                ],
            },
            "elapsed_seconds": elapsed,
        }
        return self._report

    def predict(self, df) -> np.ndarray:
        """Run the fitted preprocessing pipeline and predict on new data."""
        if self.best_model is None:
            raise ValueError("Call fit() before predict().")
        X, _, _ = self._build_features(df, transform_only=True)
        preds = self.best_model.predict(X)
        target_encoder = self._encoders.get("__target__")
        if target_encoder is not None:
            return target_encoder.inverse_transform(preds)
        return preds

    def print_report(self):
        """Pretty-print the AutoFit report."""
        if self._report is None:
            raise ValueError("Call fit() first.")
        r = self._report
        print("\n" + "=" * 60)
        print(f"  AutoFit Report — {r['task'].capitalize()}")
        print("=" * 60)
        print(f"  Target          : {r['target']}")
        print(f"  Samples / feats : {r['n_samples']} × {r['n_features']}")
        print(f"  Best model      : {r['best_model']}")
        print(f"  Best params     : {r['best_params']}")
        print(f"  Elapsed         : {r['elapsed_seconds']}s")
        print("\n── Evaluation metrics ────────────────────────────────────")
        self._pprint(r["eval_metrics"])
        print("\n── Feature importance (top 5) ────────────────────────────")
        fi = r["feature_importance"]
        for entry in fi[:5]:
            bar = "█" * int(entry["importance"] * 40)
            print(f"  {entry['feature']:<25} {bar}  {entry['importance']:.4f}")
        print()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_task(self, df) -> str:
        if self.task != "auto":
            return self.task
        dtype = df.dtypes.get(self.target, "str")
        if dtype in ("str", "bool"):
            return "classification"
        col = np.array(df[self.target], dtype=np.float64)
        unique = np.unique(col[~np.isnan(col)])
        # Treat columns with ≤ 10 unique integer values as classification
        if len(unique) <= 10 and all(v.is_integer() for v in unique):
            return "classification"
        return "regression"

    def _build_features(self, df, task=None, transform_only=False):
        """
        Impute, encode, and scale the DataFrame.
        Returns (X, y, feature_names).
        """
        features = [f for f in df.get_features() if f != self.target]
        has_target = self.target in df.get_features()
        target_col = None
        if has_target:
            if task != "classification":
                target_col = np.array(df[self.target], dtype=np.float64)
            else:
                target_col = np.array(df[self.target])
        elif not transform_only:
            raise ValueError(f"Target column '{self.target}' was not found in the DataFrame.")

        cols = []
        names = []

        for col in features:
            dtype = df.dtypes.get(col, "str")

            if dtype in ("bool", "int", "float"):
                arr = np.array(df[col], dtype=np.float64)
                # Impute
                if not transform_only:
                    imp = SimpleImputer(strategy=MedianImputer())
                    arr = imp.fit_transform(arr)
                    self._imputers[col] = imp
                else:
                    arr = self._imputers[col].transform(arr)

                # Scale (fit a per-column scaler)
                if not transform_only:
                    scaler = StandardScaler()
                    arr = scaler.fit_transform(arr)
                    self._encoders[f"__scale_{col}"] = scaler
                else:
                    arr = self._encoders[f"__scale_{col}"].transform(arr)

                cols.append(arr)
                names.append(col)

            else:
                arr = np.array(df[col], dtype=object)
                # Impute categoricals with mode
                if not transform_only:
                    imp = SimpleImputer(strategy=ModeImputer())
                    arr = imp.fit_transform(arr)
                    self._imputers[col] = imp
                else:
                    arr = self._imputers[col].transform(arr)

                # One-hot encode
                if not transform_only:
                    enc = OneHotEncoder(handle_unknown="ignore")
                    encoded = enc.fit_transform(arr)
                    self._encoders[col] = enc
                else:
                    encoded = self._encoders[col].transform(arr)

                feature_names_ohe = self._encoders[col].get_feature_names(prefix=col)
                cols.append(encoded)
                names.extend(feature_names_ohe)

        X = np.column_stack(cols) if cols else np.empty((len(df), 0))

        # Target: impute missing if any
        if target_col is None:
            return X, None, names

        if task == "classification":
            # Label-encode string targets
            if target_col.dtype == object:
                if not transform_only:
                    le = LabelEncoder()
                    target_col = le.fit_transform(target_col)
                    self._encoders["__target__"] = le
                else:
                    target_col = self._encoders["__target__"].transform(target_col)
            else:
                target_col = target_col.astype(int)
        else:
            target_col = target_col.astype(np.float64)

        return X, target_col, names

    def _split(self, X, y):
        rng = np.random.default_rng(self.seed)
        n   = len(X)
        idx = rng.permutation(n)
        n_test = max(1, int(n * self.test_size))
        test_idx  = idx[:n_test]
        train_idx = idx[n_test:]
        return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

    def _search(self, X_train, y_train, X_test, y_test, task):
        candidates = (
            _CLASSIFICATION_CANDIDATES if task == "classification"
            else _REGRESSION_CANDIDATES
        )
        metric     = accuracy if task == "classification" else \
                     (lambda a, p: -float(np.mean((a - p) ** 2)))   # neg-MSE
        greater_is_better = True

        search_results = []
        best_score     = -np.inf
        self.best_model      = None
        self.best_params     = None
        self.best_model_name = None

        for ModelClass, param_grid in candidates:
            name = ModelClass.__name__
            if self.verbose:
                print(f"  Trying {name}…")

            if not param_grid:
                # No hyperparameters to tune — just fit and score
                try:
                    model = ModelClass()
                    model.fit(X_train, y_train)
                    score = float(metric(y_test, model.predict(X_test)))
                    search_results.append({"model": name, "params": {}, "score": round(score, 4)})
                    if score > best_score:
                        best_score = score
                        self.best_model      = model
                        self.best_params     = {}
                        self.best_model_name = name
                except Exception as exc:
                    if self.verbose:
                        print(f"    {name} failed: {exc}")
                continue

            try:
                if self.search_strategy == "grid":
                    search = GridSearch(
                        ModelClass,
                        param_grid=param_grid,
                        metric=metric,
                        greater_is_better=greater_is_better,
                        cv=self.cv,
                        seed=self.seed,
                    )
                else:
                    search = RandomSearch(
                        ModelClass,
                        param_distributions=param_grid,
                        metric=metric,
                        n_iter=self.random_iter,
                        greater_is_better=greater_is_better,
                        cv=self.cv,
                        seed=self.seed,
                    )
                best_m, best_p, best_s = search.search(X_train, y_train, X_test, y_test)
                search_results.append({
                    "model":  name,
                    "params": best_p,
                    "score":  round(float(best_s), 4),
                })
                if best_s > best_score:
                    best_score = best_s
                    self.best_model      = best_m
                    self.best_params     = best_p
                    self.best_model_name = name
            except Exception as exc:
                if self.verbose:
                    print(f"    {name} failed: {exc}")

        if self.best_model is None:
            raise RuntimeError("Model search failed for every candidate.")

        return search_results

    def _get_feature_importance(self, X_train) -> list:
        """Extract feature importances from the best model if available."""
        importance_vec = None

        if hasattr(self.best_model, "feature_importance"):
            try:
                importance_vec = self.best_model.feature_importance(X_train.shape[1])
            except TypeError:
                importance_vec = self.best_model.feature_importance()

        elif hasattr(self.best_model, "weights") and self.best_model.weights is not None:
            # Linear model: use absolute weight magnitudes
            w = np.abs(self.best_model.weights)
            total = w.sum()
            importance_vec = w / total if total > 0 else w

        elif hasattr(self.best_model, "_classifiers"):
            # Logistic regression (binary)
            classifiers = self.best_model._classifiers
            if classifiers:
                w = np.abs(classifiers[0].W)
                total = w.sum()
                importance_vec = w / total if total > 0 else w

        if importance_vec is None or len(importance_vec) == 0:
            return []

        names = self._feature_names
        # Pad or trim to match
        n = min(len(names), len(importance_vec))
        result = [
            {"feature": names[i], "importance": round(float(importance_vec[i]), 4)}
            for i in range(n)
        ]
        result.sort(key=lambda x: x["importance"], reverse=True)
        return result

    @staticmethod
    def _pprint(d, indent=2):
        pad = " " * indent
        for k, v in d.items():
            if isinstance(v, dict):
                print(f"{pad}{k}:")
                AutoFit._pprint(v, indent + 2)
            else:
                print(f"{pad}{k:<20} {v}")
