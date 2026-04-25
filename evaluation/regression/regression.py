import numpy as np

def _validate(actual: np.ndarray, predicted: np.ndarray):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape:
        raise ValueError(
            f"Shape mismatch: actual {actual.shape} vs predicted {predicted.shape}"
        )
    if len(actual) == 0:
        raise ValueError("Arrays must not be empty.")
    return actual, predicted

def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Error."""
    actual, predicted = _validate(actual, predicted)
    return float(np.mean(np.abs(actual - predicted)))

def mse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Squared Error."""
    actual, predicted = _validate(actual, predicted)
    return float(np.mean((actual - predicted) ** 2))

def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(mse(actual, predicted)))

def r2(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    R² (Coefficient of Determination).
    1.0 = perfect fit. Can be negative if model is worse than a flat mean.
    """
    actual, predicted = _validate(actual, predicted)
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return float(1 - ss_res / ss_tot)

def regression_report(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Returns all regression metrics as a single dictionary."""
    return {
        "MAE":  mae(actual, predicted),
        "MSE":  mse(actual, predicted),
        "RMSE": rmse(actual, predicted),
        "R2":   r2(actual, predicted),
    }

# ──────────────────────────────────────────────
# Quick test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    y_true = np.array([3.0, -0.5, 2.0, 7.0])
    y_pred = np.array([2.5, 0.0, 2.0, 8.0])

    print("=== Regression Metrics ===")
    for name, val in regression_report(y_true, y_pred).items():
        print(f"  {name}: {val:.4f}")

    # Perfect prediction edge case
    assert r2(y_true, y_true) == 1.0
    print("\nAll regression tests passed")