import numpy as np


def mean(values: np.ndarray) -> np.float64:
    return np.mean(values)


def std(values: np.ndarray) -> np.float64:
    return np.std(values, ddof=1)  # sample std


def median(values: np.ndarray) -> np.float64:
    return np.median(values)


def mode(values: np.ndarray):
    unique_values, counts = np.unique(values, return_counts=True)
    return unique_values[np.argmax(counts)]


def count(values: np.ndarray) -> np.int64:
    return np.count_nonzero(~np.isnan(values))


def quantile(values: np.ndarray, percentage: float) -> np.float64:
    return np.quantile(values, percentage)


def col_min(values: np.ndarray) -> np.float64:
    return np.min(values)


def col_max(values: np.ndarray) -> np.float64:
    return np.max(values)


def skewness(values: np.ndarray) -> np.float64:
    mu    = np.mean(values)
    sigma = np.std(values, ddof=0)
    if sigma == 0:
        return np.float64(0.0)
    return np.mean(((values - mu) / sigma) ** 3)


def kurtosis(values: np.ndarray) -> np.float64:
    mu    = np.mean(values)
    sigma = np.std(values, ddof=0)
    if sigma == 0:
        return np.float64(0.0)
    return np.mean(((values - mu) / sigma) ** 4) - 3


def pearson_correlation(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    n_samples, n_features = X.shape

    mu    = np.mean(X, axis=0)
    sigma = np.std(X, axis=0, ddof=0)

    X_centered = X - mu
    corr = np.zeros((n_features, n_features), dtype=np.float64)

    for i in range(n_features):
        for j in range(n_features):
            denom = sigma[i] * sigma[j]
            if denom == 0:
                corr[i, j] = 0.0
            else:
                corr[i, j] = np.mean(X_centered[:, i] * X_centered[:, j]) / denom

    return corr


# =========================
# MISSING VALUES HANDLING
# =========================

def is_missing(x):
    return isinstance(x, (float, np.floating)) and np.isnan(x)


# =========================
# DISTANCE METRICS
# =========================

def euclidean_distance(x: np.ndarray, y: np.ndarray) -> np.float64:
    """
    Euclidean (L2) distance between two mixed-type vectors.
    Missing values on either side are skipped.
    Returns np.inf if no valid dimensions remain.
    """
    dis = 0.0
    count_valid = 0

    for i in range(len(x)):
        v1, v2 = x[i], y[i]
        if is_missing(v1) or is_missing(v2):
            continue
        if isinstance(v1, (int, float, np.floating)) and isinstance(v2, (int, float, np.floating)):
            dis += (v1 - v2) ** 2
        else:
            dis += 0 if v1 == v2 else 1
        count_valid += 1

    if count_valid == 0:
        return np.float64(np.inf)
    return np.float64(np.sqrt(dis))


def manhattan_distance(x: np.ndarray, y: np.ndarray) -> np.float64:
    """
    Manhattan (L1) distance between two mixed-type vectors.
    Missing values on either side are skipped.
    Returns np.inf if no valid dimensions remain.
    """
    dis = 0.0
    count_valid = 0

    for i in range(len(x)):
        v1, v2 = x[i], y[i]
        if is_missing(v1) or is_missing(v2):
            continue
        if isinstance(v1, (int, float, np.floating)) and isinstance(v2, (int, float, np.floating)):
            dis += abs(v1 - v2)
        else:
            dis += 0 if v1 == v2 else 1
        count_valid += 1

    if count_valid == 0:
        return np.float64(np.inf)
    return np.float64(dis)


def distance(
    x: np.ndarray,
    y: np.ndarray,
    metric: str = "euclidean",
) -> np.float64:
    """
    Unified distance function.

    Parameters
    ----------
    x, y   : array-like — mixed-type vectors (numerical or categorical)
    metric : str        — 'euclidean' (default) | 'manhattan'
    """
    if metric == "euclidean":
        return euclidean_distance(x, y)
    if metric == "manhattan":
        return manhattan_distance(x, y)
    raise ValueError(f"Unknown metric '{metric}'. Use 'euclidean' or 'manhattan'.")


def majority_vote(values):
    """Returns the most frequent value. Works with lists or arrays."""
    values = list(values)
    unique_values, counts = np.unique(values, return_counts=True)
    return unique_values[np.argmax(counts)]