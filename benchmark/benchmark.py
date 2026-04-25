"""
GlassBox Benchmark
==================
Compares GlassBox AutoFit against Scikit-Learn RandomForest
on three standard datasets:
  - Iris        (classification, small, clean)
  - Titanic     (classification, medium, missing values)
  - Boston      (regression, medium, clean)

Run from project root:
    python -m benchmark.benchmark
"""

import sys
import numpy as np
from io import StringIO

import pandas as pd
from sklearn.ensemble import RandomForestClassifier as SklearnRFC
from sklearn.ensemble import RandomForestRegressor as SklearnRFR
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder as SklearnLE

from dataframes.DataFrame import DataFrame
from pipeline.autofit import AutoFit
from models.logisticregression.logistic_regression import LogisticRegression
from models.linearregression.linear import LinearRegression
from models.naive.naive import NaiveBayes
from models.decisiontree.decisiontreeclassifier import DecisionTreeClassifier
from models.decisiontree.decisiontreeregression import DecisionTreeRegression


# ── Fast candidate sets (no KNN, no RandomForest) ──────────────
# KNN is O(n²) in pure NumPy — too slow for 500+ row datasets
# RandomForest with 50-100 trees is also very slow in pure NumPy

FAST_CLASSIFICATION = [
    (LogisticRegression,     {"lr": [0.05, 0.1], "max_iters": [500]}),
    (DecisionTreeClassifier, {"max_depth": [3, 5]}),
    (NaiveBayes,             {}),
]

FAST_REGRESSION = [
    (LinearRegression,       {"lr": [0.01], "epochs": [1000]}),
    (DecisionTreeRegression, {"max_depth": [3, 5]}),
]


# ── Helpers ─────────────────────────────────────────────────────

def load_glassbox(csv_path, drop_cols=None):
    """Load a CSV into GlassBox DataFrame, optionally dropping columns."""
    if drop_cols:
        pdf = pd.read_csv(csv_path)
        pdf = pdf.drop(columns=drop_cols, errors="ignore")
        return DataFrame.load_csv(StringIO(pdf.to_csv(index=False)))
    return DataFrame.load_csv(csv_path)


def sklearn_encode(pdf, target):
    """Label-encode all object columns for Scikit-Learn."""
    X = pdf.drop(columns=[target]).copy()
    y = pdf[target].copy()
    for col in X.select_dtypes(include="object").columns:
        X[col] = SklearnLE().fit_transform(X[col].astype(str))
    if y.dtype == object:
        y = SklearnLE().fit_transform(y)
    return X, y


# ── Core benchmark functions ────────────────────────────────────

def benchmark_classification(csv_path, target, drop_cols=None, fast=False):
    print(f"\n{'='*60}")
    print(f"  Dataset : {csv_path}")
    print(f"  Target  : {target}")
    print(f"  Mode    : {'fast — LogReg + DecTree + NaiveBayes only' if fast else 'full'}")
    print(f"{'='*60}")

    # ── GlassBox ──
    print("\n[GlassBox]")
    df = load_glassbox(csv_path, drop_cols)
    report = AutoFit(
        target=target,
        task="classification",
        cv=3,
        search_strategy="random",
        random_iter=5,
        verbose=True,
        candidates=FAST_CLASSIFICATION if fast else None,
    ).fit(df)

    glassbox_acc = report["eval_metrics"]["accuracy"]
    print(f"  Best model : {report['best_model']}")
    print(f"  Accuracy   : {glassbox_acc:.4f}")

    # ── Scikit-Learn ──
    print("\n[Scikit-Learn RandomForest baseline]")
    pdf = pd.read_csv(csv_path)
    if drop_cols:
        pdf = pdf.drop(columns=drop_cols, errors="ignore")
    pdf = pdf.dropna()
    X, y = sklearn_encode(pdf, target)
    sklearn_scores = cross_val_score(
        SklearnRFC(n_estimators=100, random_state=42),
        X, y, cv=3, scoring="accuracy"
    )
    sklearn_acc = float(sklearn_scores.mean())
    print(f"  Accuracy   : {sklearn_acc:.4f}")

    # ── Result ──
    ratio = glassbox_acc / sklearn_acc if sklearn_acc > 0 else 0.0
    status = "✅ PASS" if ratio >= 0.90 else "❌ FAIL"
    print(f"\n  Ratio (GlassBox / Sklearn) : {ratio:.4f}  {status}")

    return {
        "dataset":          csv_path,
        "target":           target,
        "glassbox":         round(glassbox_acc, 4),
        "sklearn":          round(sklearn_acc, 4),
        "ratio":            round(ratio, 4),
        "status":           status,
    }


def benchmark_regression(csv_path, target, drop_cols=None, fast=False):
    print(f"\n{'='*60}")
    print(f"  Dataset : {csv_path}")
    print(f"  Target  : {target}")
    print(f"  Mode    : {'fast — LinReg + DecTree only' if fast else 'full'}")
    print(f"{'='*60}")

    # ── GlassBox ──
    print("\n[GlassBox]")
    df = load_glassbox(csv_path, drop_cols)
    report = AutoFit(
        target=target,
        task="regression",
        cv=3,
        search_strategy="random",
        random_iter=5,
        verbose=True,
        candidates=FAST_REGRESSION if fast else None,
    ).fit(df)

    glassbox_r2 = report["eval_metrics"]["R2"]
    print(f"  Best model : {report['best_model']}")
    print(f"  R²         : {glassbox_r2:.4f}")

    # ── Scikit-Learn ──
    print("\n[Scikit-Learn RandomForest baseline]")
    pdf = pd.read_csv(csv_path)
    if drop_cols:
        pdf = pdf.drop(columns=drop_cols, errors="ignore")
    pdf = pdf.dropna()
    X, y = sklearn_encode(pdf, target)
    sklearn_scores = cross_val_score(
        SklearnRFR(n_estimators=100, random_state=42),
        X, y, cv=3, scoring="r2"
    )
    sklearn_r2 = float(sklearn_scores.mean())
    print(f"  R²         : {sklearn_r2:.4f}")

    # ── Result ──
    ratio = glassbox_r2 / sklearn_r2 if sklearn_r2 > 0 else 0.0
    status = "✅ PASS" if ratio >= 0.90 else "❌ FAIL"
    print(f"\n  Ratio (GlassBox / Sklearn) : {ratio:.4f}  {status}")

    return {
        "dataset":      csv_path,
        "target":       target,
        "glassbox_r2":  round(glassbox_r2, 4),
        "sklearn_r2":   round(sklearn_r2, 4),
        "ratio":        round(ratio, 4),
        "status":       status,
    }


# ── Main ────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = []

    # Iris — small dataset, safe to run full model set
    results.append(benchmark_classification(
        "iris.csv",
        target="class",
        fast=False,
    ))

    # Titanic — large dataset, skip KNN and RandomForest
    results.append(benchmark_classification(
        "titanic.csv",
        target="Survived",
        drop_cols=["Name", "Ticket", "Cabin", "PassengerId"],
        fast=True,
    ))

    # Boston — large dataset, skip KNN and RandomForest
    results.append(benchmark_regression(
        "boston.csv",
        target="medv",
        fast=True,
    ))

    # ── Final summary ──
    print(f"\n{'='*60}")
    print("  FINAL BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Dataset':<20} {'GlassBox':>10} {'Sklearn':>10} {'Ratio':>8}  Status")
    print(f"  {'-'*56}")
    for r in results:
        gb  = r.get("glassbox", r.get("glassbox_r2", "?"))
        sk  = r.get("sklearn",  r.get("sklearn_r2",  "?"))
        print(f"  {str(r['dataset']):<20} {gb:>10.4f} {sk:>10.4f} {r['ratio']:>8.4f}  {r['status']}")
    print()

    # Exit with error code if any test failed
    failed = [r for r in results if r["status"] != "✅ PASS"]
    if failed:
        print(f"  ⚠️  {len(failed)} benchmark(s) failed.")
        sys.exit(1)
    else:
        print("  All benchmarks passed ≥90% of Scikit-Learn baseline.")
        sys.exit(0)