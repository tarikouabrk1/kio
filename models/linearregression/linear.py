import numpy as np

# =========================================================
# Learning Rate Schedules
# =========================================================

def _constant_lr(lr0: float, epoch: int) -> float:
    return lr0


def _step_decay_lr(lr0: float, epoch: int,
                   drop: float = 0.5,
                   epochs_drop: int = 100) -> float:
    return lr0 * (drop ** (epoch // epochs_drop))


def _cosine_decay_lr(lr0: float, epoch: int,
                     total_epochs: int) -> float:
    return lr0 * 0.5 * (1 + np.cos(np.pi * epoch / total_epochs))


SCHEDULES = {
    "constant": _constant_lr,
    "step": _step_decay_lr,
    "cosine": _cosine_decay_lr,
}


# =========================================================
# Linear Regression (Gradient Descent)
# =========================================================

class LinearRegression:

    def __init__(
        self,
        lr: float = 0.01,
        epochs: int = 1000,
        schedule: str = "constant",
        drop: float = 0.5,
        epochs_drop: int = 100,
        normalize: bool = False,
        tol: float = 1e-6,
        verbose: bool = False,
        clip_grad: float = None  # New parameter
    ):
        if schedule not in SCHEDULES:
            raise ValueError(f"schedule must be one of {list(SCHEDULES)}")

        self.lr0 = lr
        self.epochs = epochs
        self.schedule = schedule

        # step decay params
        self.drop = drop
        self.epochs_drop = epochs_drop

        # training options
        self.normalize = normalize
        self.tol = tol
        self.verbose = verbose
        self.clip_grad = clip_grad

        # model params
        self.weights: np.ndarray | None = None
        self.bias: float | None = None

        # normalization params
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

        self.loss_history: list[float] = []
        self.n_epochs_trained: int = 0

    # =====================================================
    # Internal helpers
    # =====================================================

    def _get_lr(self, epoch: int) -> float:
        func = SCHEDULES[self.schedule]

        if self.schedule == "step":
            return func(self.lr0, epoch, self.drop, self.epochs_drop)
        elif self.schedule == "cosine":
            return func(self.lr0, epoch, self.epochs)
        return func(self.lr0, epoch)

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        if self.mean is None:
            self.mean = X.mean(axis=0)
            self.std = X.std(axis=0)
            self.std[self.std < 1e-8] = 1.0

        return (X - self.mean) / self.std

    # =====================================================
    # Training
    # =====================================================

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        # Input validation
        if X.size == 0 or y.size == 0:
            raise ValueError("Empty training data")
        if len(X) != len(y):
            raise ValueError(f"X and y have mismatched lengths: {len(X)} vs {len(y)}")
        if np.any(np.isnan(X)) or np.any(np.isnan(y)):
            raise ValueError("Input data contains NaN values")
        if np.any(np.isinf(X)) or np.any(np.isinf(y)):
            raise ValueError("Input data contains Inf values")

        if self.normalize:
            X = self._normalize(X)

        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []

        # Initialize epoch variable
        epoch = 0
        
        for epoch in range(self.epochs):

            lr = self._get_lr(epoch)

            # forward pass
            y_pred = X @ self.weights + self.bias
            error = y_pred - y

            # compute loss BEFORE update (saves one forward pass)
            mse = np.mean(error ** 2)
            self.loss_history.append(mse)

            # gradients
            dw = (2.0 / n_samples) * X.T @ error
            db = (2.0 / n_samples) * error.sum()

            # gradient clipping (if enabled)
            if self.clip_grad is not None:
                dw = np.clip(dw, -self.clip_grad, self.clip_grad)
                db = np.clip(db, -self.clip_grad, self.clip_grad)

            # update
            self.weights -= lr * dw
            self.bias -= lr * db

            # verbose logging
            if self.verbose and epoch % 100 == 0:
                print(f"Epoch {epoch} | Loss: {mse:.6f} | LR: {lr:.6f}")

            # early stopping
            if epoch > 0:
                loss_change = abs(self.loss_history[-1] - self.loss_history[-2])
                if loss_change < self.tol:
                    if self.verbose:
                        print(f"Converged at epoch {epoch} (Δloss: {loss_change:.2e})")
                    break

        self.n_epochs_trained = epoch + 1
        return self

    # =====================================================
    # Prediction
    # =====================================================

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None or self.bias is None:
            raise ValueError("Model not fitted yet.")

        X = np.asarray(X, dtype=float)

        if self.normalize:
            if self.mean is None or self.std is None:
                raise ValueError("Normalization parameters not found. Fit the model first.")
            X = (X - self.mean) / self.std

        return X @ self.weights + self.bias

    # =====================================================
    # Scoring
    # =====================================================

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """R^2 coefficient of determination"""
        y = np.asarray(y, dtype=float)
        y_pred = self.predict(X)

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0

        return 1 - (ss_res / ss_tot)

    # =====================================================
    # Model information
    # =====================================================

    def get_params(self) -> dict:
        return {
            'weights': self.weights.copy() if self.weights is not None else None,
            'bias': float(self.bias) if self.bias is not None else None,
            'mean': self.mean.copy() if self.mean is not None else None,
            'std': self.std.copy() if self.std is not None else None,
            'epochs_trained': self.n_epochs_trained
        }
