import numpy as np


# =========================================================
# Learning rate schedules (mirrors LinearRegression)
# =========================================================

def _constant_lr(lr0, epoch, **_):
    return lr0

def _step_decay_lr(lr0, epoch, drop=0.5, epochs_drop=100, **_):
    return lr0 * (drop ** (epoch // epochs_drop))

def _cosine_decay_lr(lr0, epoch, total_epochs=1000, **_):
    return lr0 * 0.5 * (1 + np.cos(np.pi * epoch / total_epochs))

SCHEDULES = {
    "constant": _constant_lr,
    "step":     _step_decay_lr,
    "cosine":   _cosine_decay_lr,
}


# =========================================================
# Binary logistic classifier (used internally)
# =========================================================

class _BinaryLogistic:
    def __init__(self, lr, max_iters, schedule, drop, epochs_drop,
                 tol, l2, clip_grad, verbose):
        self.lr0 = lr
        self.max_iters = max_iters
        self.schedule = schedule
        self.drop = drop
        self.epochs_drop = epochs_drop
        self.tol = tol
        self.l2 = l2
        self.clip_grad = clip_grad
        self.verbose = verbose
        self.W = None
        self.b = None
        self.loss_history = []

    @staticmethod
    def _sigmoid(z):
        return np.where(z >= 0,
                        1 / (1 + np.exp(-z)),
                        np.exp(z) / (1 + np.exp(z)))

    def _get_lr(self, epoch):
        fn = SCHEDULES[self.schedule]
        return fn(self.lr0, epoch,
                  drop=self.drop,
                  epochs_drop=self.epochs_drop,
                  total_epochs=self.max_iters)

    def fit(self, X, y):
        n, m = X.shape
        self.W = np.zeros(m)
        self.b = 0.0
        self.loss_history = []
        epoch = 0

        for epoch in range(self.max_iters):
            lr = self._get_lr(epoch)
            z = X @ self.W + self.b
            y_pred = self._sigmoid(z)

            # Binary cross-entropy
            eps = 1e-12
            loss = -np.mean(y * np.log(y_pred + eps) +
                            (1 - y) * np.log(1 - y_pred + eps))
            loss += 0.5 * self.l2 * np.sum(self.W ** 2)
            self.loss_history.append(float(loss))

            dw = (1 / n) * (X.T @ (y_pred - y)) + self.l2 * self.W
            db = (1 / n) * np.sum(y_pred - y)

            if self.clip_grad is not None:
                dw = np.clip(dw, -self.clip_grad, self.clip_grad)
                db = float(np.clip(db, -self.clip_grad, self.clip_grad))

            self.W -= lr * dw
            self.b -= lr * db

            if self.verbose and epoch % 100 == 0:
                print(f"Epoch {epoch} | Loss: {loss:.6f} | LR: {lr:.6f}")

            if epoch > 0 and abs(self.loss_history[-1] - self.loss_history[-2]) < self.tol:
                break

        return self

    def predict_proba(self, X):
        return self._sigmoid(X @ self.W + self.b)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


# =========================================================
# Public LogisticRegression (binary + multiclass OvR)
# =========================================================

class LogisticRegression:
    """
    Logistic Regression via gradient descent.

    Supports binary and multiclass (One-vs-Rest) classification.

    Parameters
    ----------
    lr          : float  — initial learning rate
    max_iters   : int    — maximum gradient descent iterations
    schedule    : str    — 'constant' | 'step' | 'cosine'
    drop        : float  — step decay drop factor
    epochs_drop : int    — step decay period
    tol         : float  — early-stopping tolerance on loss change
    l2          : float  — L2 regularisation strength (0 = disabled)
    clip_grad   : float  — gradient clipping value (None = disabled)
    verbose     : bool   — print loss every 100 epochs
    """

    def __init__(
        self,
        lr: float = 0.1,
        max_iters: int = 1000,
        schedule: str = "constant",
        drop: float = 0.5,
        epochs_drop: int = 100,
        tol: float = 1e-6,
        l2: float = 0.0,
        clip_grad: float = None,
        verbose: bool = False,
    ):
        if schedule not in SCHEDULES:
            raise ValueError(f"schedule must be one of {list(SCHEDULES)}")
        self.lr = lr
        self.max_iters = max_iters
        self.schedule = schedule
        self.drop = drop
        self.epochs_drop = epochs_drop
        self.tol = tol
        self.l2 = l2
        self.clip_grad = clip_grad
        self.verbose = verbose

        self.classes_ = None
        self._classifiers = []     # one per class in OvR

    # ---- helpers ------------------------------------------------

    def _make_binary(self):
        return _BinaryLogistic(
            lr=self.lr, max_iters=self.max_iters,
            schedule=self.schedule, drop=self.drop,
            epochs_drop=self.epochs_drop, tol=self.tol,
            l2=self.l2, clip_grad=self.clip_grad, verbose=self.verbose,
        )

    @staticmethod
    def _validate(X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        if X.size == 0 or y.size == 0:
            raise ValueError("Empty training data.")
        if X.ndim != 2:
            raise ValueError("X must be a 2-D array.")
        if y.ndim != 1:
            raise ValueError("y must be a 1-D array.")
        if len(X) != len(y):
            raise ValueError(f"X and y length mismatch: {len(X)} vs {len(y)}")
        if np.any(np.isnan(X)):
            raise ValueError("X contains NaN values.")
        return X, y

    # ---- public API ---------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray):
        X, y = self._validate(X, y)
        self.classes_ = np.unique(y)
        self._classifiers = []

        if len(self.classes_) == 2:
            # Binary: treat second class as positive
            clf = self._make_binary()
            y_bin = (y == self.classes_[1]).astype(np.float64)
            clf.fit(X, y_bin)
            self._classifiers.append(clf)
        else:
            # One-vs-Rest multiclass
            for c in self.classes_:
                clf = self._make_binary()
                y_bin = (y == c).astype(np.float64)
                clf.fit(X, y_bin)
                self._classifiers.append(clf)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._classifiers:
            raise ValueError("Model not fitted yet.")
        X = np.asarray(X, dtype=np.float64)

        if len(self.classes_) == 2:
            pos_prob = self._classifiers[0].predict_proba(X)
            return np.column_stack([1 - pos_prob, pos_prob])

        # OvR: stack raw scores and normalise
        scores = np.column_stack(
            [clf.predict_proba(X) for clf in self._classifiers]
        )
        row_sums = scores.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        return scores / row_sums

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        return self.classes_[indices]

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Alias for predict — satisfies transformer interface."""
        return self.predict(X)

    # Weight access for binary case
    def get_W(self):
        if len(self._classifiers) == 1:
            return self._classifiers[0].W
        return [clf.W for clf in self._classifiers]

    def get_b(self):
        if len(self._classifiers) == 1:
            return self._classifiers[0].b
        return [clf.b for clf in self._classifiers]