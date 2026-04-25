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
    mu = np.mean(values)
    sigma = np.std(values, ddof=0)

    if sigma == 0:
        return np.float64(0.0)

    return np.mean(((values - mu) / sigma) ** 3)


def kurtosis(values: np.ndarray) -> np.float64:
    mu = np.mean(values)
    sigma = np.std(values, ddof=0)

    if sigma == 0:
        return np.float64(0.0)

    return np.mean(((values - mu) / sigma) ** 4) - 3


def pearson_correlation(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    n_samples, n_features = X.shape

    mu = np.mean(X, axis=0)
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
# DISTANCE (Mixed data)
# =========================

def distance(x: np.ndarray, y: np.ndarray) -> np.float64:
    dis = 0.0
    count_valid = 0

    for i in range(len(x)):
        val1 = x[i]
        val2 = y[i]

        if is_missing(val1) or is_missing(val2):
            continue

        # numerical
        if isinstance(val1, (int, float, np.floating)) and isinstance(val2, (int, float, np.floating)):
            dis += (val1 - val2) ** 2
        else:
            # categorical
            dis += 0 if val1 == val2 else 1

        count_valid += 1

    if count_valid == 0:
        return np.float64(np.inf)

    return np.sqrt(dis)


def majority_vote(values):
    """
    Returns the most frequent value.
    Works with lists or arrays.
    """
    values = list(values)
    unique_values, counts = np.unique(values, return_counts=True)
    return unique_values[np.argmax(counts)]
