import numpy as np

def _validate(actual: np.ndarray, predicted: np.ndarray):
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    if actual.shape != predicted.shape:
        raise ValueError(
            f"Shape mismatch: actual {actual.shape} vs predicted {predicted.shape}"
        )
    if len(actual) == 0:
        raise ValueError("Arrays must not be empty.")
    return actual, predicted

def confusion_matrix(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """
    Builds an N×N confusion matrix for N classes.

    Rows = actual class, Columns = predicted class.
    Entry [i, j] = number of samples of class i predicted as class j.
    """
    actual, predicted = _validate(actual, predicted)
    classes = np.unique(np.concatenate([actual, predicted]))
    n = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    matrix = np.zeros((n, n), dtype=int)
    for a, p in zip(actual, predicted):
        matrix[class_to_idx[a], class_to_idx[p]] += 1

    return matrix

def print_confusion_matrix(actual: np.ndarray,predicted: np.ndarray,labels: list = None):
    """Pretty-prints the confusion matrix."""
    actual, predicted = _validate(actual, predicted)
    classes = np.unique(np.concatenate([actual, predicted]))
    matrix = confusion_matrix(actual, predicted)

    label_names = [str(labels[i]) if labels else str(c) for i, c in enumerate(classes)]
    col_width = max(len(n) for n in label_names) + 2

    header = " " * col_width + "".join(n.center(col_width) for n in label_names)
    print("\nConfusion Matrix (rows=actual, cols=predicted)")
    print(header)
    print("-" * len(header))
    for i, row_label in enumerate(label_names):
        row = row_label.ljust(col_width)
        row += "".join(str(matrix[i, j]).center(col_width) for j in range(len(label_names)))
        print(row)
    print()

def _binary_counts(actual, predicted, pos_label):
    tp = int(np.sum((actual == pos_label) & (predicted == pos_label)))
    fp = int(np.sum((actual != pos_label) & (predicted == pos_label)))
    fn = int(np.sum((actual == pos_label) & (predicted != pos_label)))
    tn = int(np.sum((actual != pos_label) & (predicted != pos_label)))
    return tp, fp, fn, tn

def accuracy(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Fraction of correctly classified samples."""
    actual, predicted = _validate(actual, predicted)
    return float(np.mean(actual == predicted))

def precision(actual: np.ndarray,predicted: np.ndarray,pos_label=1,average: str = "binary") -> float:
    """
    Precision = TP / (TP + FP).
    average:
        'binary'  – single positive class (pos_label).
        'macro'   – unweighted mean across all classes.
        'weighted'– mean weighted by support (class frequency).
    """
    actual, predicted = _validate(actual, predicted)

    if average == "binary":
        tp, fp, fn, tn = _binary_counts(actual, predicted, pos_label)
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    classes = np.unique(actual)
    scores, weights = [], []
    for c in classes:
        tp, fp, fn, tn = _binary_counts(actual, predicted, c)
        scores.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
        weights.append(int(np.sum(actual == c)))

    if average == "macro":
        return float(np.mean(scores))
    if average == "weighted":
        return float(np.average(scores, weights=weights))

    raise ValueError("average must be 'binary', 'macro', or 'weighted'.")


def recall(actual: np.ndarray,predicted: np.ndarray,pos_label=1,average: str = "binary") -> float:
    """
    Recall = TP / (TP + FN).
    average: same options as precision().
    """
    actual, predicted = _validate(actual, predicted)

    if average == "binary":
        tp, fp, fn, tn = _binary_counts(actual, predicted, pos_label)
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    classes = np.unique(actual)
    scores, weights = [], []
    for c in classes:
        tp, fp, fn, tn = _binary_counts(actual, predicted, c)
        scores.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        weights.append(int(np.sum(actual == c)))

    if average == "macro":
        return float(np.mean(scores))
    if average == "weighted":
        return float(np.average(scores, weights=weights))

    raise ValueError("average must be 'binary', 'macro', or 'weighted'.")


def f1_score(actual: np.ndarray,predicted: np.ndarray,pos_label=1,average: str = "binary") -> float:
    """
    F1 = 2 * (precision * recall) / (precision + recall).
    Harmonic mean of precision and recall.
    average: same options as precision().
    """
    p = precision(actual, predicted, pos_label=pos_label, average=average)
    r = recall(actual, predicted, pos_label=pos_label, average=average)
    return float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0


def classification_report(actual: np.ndarray,predicted: np.ndarray,labels: list = None) -> dict:
    """
    Returns a per-class breakdown + overall metrics.
    Structure:
    {
        "classes": {
            "0": {"precision": ..., "recall": ..., "f1": ..., "support": ...},
            ...
        },
        "accuracy": ...,
        "macro avg":    {"precision": ..., "recall": ..., "f1": ...},
        "weighted avg": {"precision": ..., "recall": ..., "f1": ...},
    }
    """
    actual, predicted = _validate(actual, predicted)
    classes = np.unique(np.concatenate([actual, predicted]))

    report = {"classes": {}}

    for c in classes:
        tp, fp, fn, tn = _binary_counts(actual, predicted, c)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        support = int(np.sum(actual == c))
        label = str(labels[list(classes).index(c)]) if labels else str(c)
        report["classes"][label] = {
            "precision": round(p, 4),
            "recall":    round(r, 4),
            "f1":        round(f, 4),
            "support":   support,
        }

    report["accuracy"]     = round(accuracy(actual, predicted), 4)
    report["macro avg"]    = {
        "precision": round(precision(actual, predicted, average="macro"), 4),
        "recall":    round(recall(actual, predicted, average="macro"), 4),
        "f1":        round(f1_score(actual, predicted, average="macro"), 4),
    }
    report["weighted avg"] = {
        "precision": round(precision(actual, predicted, average="weighted"), 4),
        "recall":    round(recall(actual, predicted, average="weighted"), 4),
        "f1":        round(f1_score(actual, predicted, average="weighted"), 4),
    }

    return report


def print_classification_report(actual: np.ndarray,predicted: np.ndarray,labels: list = None):
    """Pretty-prints the classification report."""
    report = classification_report(actual, predicted, labels)

    print(f"\n{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 55)
    for cls, metrics in report["classes"].items():
        print(
            f"{cls:<15} {metrics['precision']:>10.4f}"
            f" {metrics['recall']:>10.4f}"
            f" {metrics['f1']:>10.4f}"
            f" {metrics['support']:>10}"
        )
    print("-" * 55)
    print(f"{'Accuracy':<15} {'':>10} {'':>10} {report['accuracy']:>10.4f}")
    for avg in ("macro avg", "weighted avg"):
        m = report[avg]
        print(
            f"{avg:<15} {m['precision']:>10.4f}"
            f" {m['recall']:>10.4f}"
            f" {m['f1']:>10.4f}"
        )
    print()

# ──────────────────────────────────────────────
# Quick test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 1])

    print("=== Binary Classification ===")
    print_confusion_matrix(y_true, y_pred, labels=["Negative", "Positive"])
    print_classification_report(y_true, y_pred, labels=["Negative", "Positive"])

    # Multiclass
    y_true_mc = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    y_pred_mc = np.array([0, 2, 2, 0, 0, 2, 1, 1, 0])

    print("=== Multiclass Classification ===")
    print_confusion_matrix(y_true_mc, y_pred_mc)
    print_classification_report(y_true_mc, y_pred_mc)

    # Edge: perfect predictions
    assert accuracy(y_true, y_true) == 1.0
    assert f1_score(y_true, y_true, average="macro") == 1.0

    print("All classification tests passed")