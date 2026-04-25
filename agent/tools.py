"""
GlassBox MCP Tool Wrapper
=========================
Exposes GlassBox-AutoML as a callable "Skill" for IronClaw (NEAR AI) agents.

The agent calls `automl_tool(csv_text, target, task)` and receives a
structured JSON report it can read aloud / explain to the user.

All heavy work runs inside the standard Python (or Pyodide/WASM) runtime —
no external dependencies beyond NumPy.

Tool schema
-----------
{
  "name": "automl_tool",
  "description": "Runs automated EDA, preprocessing, model search, and
                  evaluation on a CSV dataset.",
  "parameters": {
    "csv_text":  { "type": "string",  "description": "Raw CSV content." },
    "target":    { "type": "string",  "description": "Target column name." },
    "task":      { "type": "string",  "enum": ["auto","classification",
                                               "regression"],
                   "default": "auto" }
  }
}
"""

import json
import numpy as np

from dataframes.DataFrame import DataFrame
from pipeline.autofit import AutoFit


# =========================================================
# Public tool function
# =========================================================

def automl_tool(
    csv_text: str,
    target: str,
    task: str = "auto",
    cv: int = 3,
    test_size: float = 0.2,
    seed: int = 42,
    search_strategy: str = "grid",
    random_iter: int = 10,
    verbose: bool = False,
) -> str:
    """
    Entry point called by the IronClaw agent.

    Parameters
    ----------
    csv_text  : str   — raw CSV string (first row = header)
    target    : str   — name of the column to predict
    task      : str   — 'auto' | 'classification' | 'regression'
    cv        : int   — cross-validation folds
    test_size : float — hold-out fraction
    seed      : int   — random seed
    verbose   : bool  — print progress to stdout

    Returns
    -------
    str — JSON report (UTF-8 encoded)
    """
    try:
        df = _load_csv_from_string(csv_text)
        pipeline = AutoFit(
            target=target,
            task=task,
            cv=cv,
            test_size=test_size,
            seed=seed,
            search_strategy=search_strategy,
            random_iter=random_iter,
            verbose=verbose,
        )
        report = pipeline.fit(df)
        # Attach a plain-English summary the agent can speak
        report["agent_summary"] = _build_summary(report)
        return json.dumps(report, default=_json_safe, indent=2)

    except Exception as exc:
        error_payload = {
            "status":  "error",
            "message": str(exc),
            "agent_summary": f"The AutoML pipeline failed: {exc}",
        }
        return json.dumps(error_payload, indent=2)


# =========================================================
# EDA-only tool (lighter, faster)
# =========================================================

def eda_tool(csv_text: str) -> str:
    """
    Runs only the EDA Inspector and returns a JSON report.
    Useful when the agent only needs a data audit.
    """
    try:
        from eda.inspector import EDAInspector
        df     = _load_csv_from_string(csv_text)
        report = EDAInspector(df).run()
        return json.dumps(report, default=_json_safe, indent=2)
    except Exception as exc:
        return json.dumps({"status": "error", "message": str(exc)}, indent=2)


# =========================================================
# Internal helpers
# =========================================================

def _load_csv_from_string(csv_text: str) -> DataFrame:
    """Load csv_text directly from memory for sandbox- and WASM-friendly use."""
    from io import StringIO

    return DataFrame.load_csv(StringIO(csv_text))


def _json_safe(obj):
    """JSON serialiser for NumPy scalars."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable.")


def _build_summary(report: dict) -> str:
    """Compose a short natural-language summary for the agent to surface."""
    task    = report.get("task", "unknown")
    target  = report.get("target", "the target")
    model   = report.get("best_model", "unknown model")
    params  = report.get("best_params", {})
    metrics = report.get("eval_metrics", {})
    fi      = report.get("feature_importance", [])
    top_f   = fi[0]["feature"] if fi else "unknown"
    elapsed = report.get("elapsed_seconds", "?")

    # Pick the headline metric
    if task == "classification":
        score_val = metrics.get("accuracy", "N/A")
        score_str = f"accuracy of {score_val:.2%}" if isinstance(score_val, float) else str(score_val)
    else:
        r2 = metrics.get("R2", "N/A")
        score_str = f"R² of {r2:.4f}" if isinstance(r2, float) else str(r2)

    param_str = (
        ", ".join(f"{k}={v}" for k, v in params.items())
        if params else "default settings"
    )

    return (
        f"I trained a {task} model to predict '{target}'. "
        f"After searching {len(report.get('search_results', []))} model types, "
        f"the best was {model} (with {param_str}), "
        f"achieving a {score_str} on the held-out test set. "
        f"The most important feature was '{top_f}'. "
        f"The full pipeline finished in {elapsed} seconds."
    )


# =========================================================
# Tool schema (for IronClaw / MCP registration)
# =========================================================

TOOL_SCHEMA = {
    "tools": [
        {
            "name": "automl_tool",
            "description": (
                "Runs an end-to-end AutoML pipeline (EDA → Preprocessing → "
                "Model Search → Evaluation) on a CSV dataset and returns a "
                "structured JSON report including the best model, evaluation "
                "metrics, and feature importances."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "csv_text":  {
                        "type":        "string",
                        "description": "Raw CSV content as a string (header row required).",
                    },
                    "target": {
                        "type":        "string",
                        "description": "Name of the column to predict.",
                    },
                    "task": {
                        "type":        "string",
                        "enum":        ["auto", "classification", "regression"],
                        "default":     "auto",
                        "description": "ML task type. 'auto' infers from the target column.",
                    },
                    "search_strategy": {
                        "type":        "string",
                        "enum":        ["grid", "random"],
                        "default":     "grid",
                        "description": "Hyperparameter search strategy.",
                    },
                    "random_iter": {
                        "type":        "integer",
                        "default":     10,
                        "description": "Number of random search samples when search_strategy='random'.",
                    },
                },
                "required": ["csv_text", "target"],
            },
        },
        {
            "name": "eda_tool",
            "description": (
                "Runs only the EDA Inspector on a CSV dataset. Returns a JSON "
                "report with column profiles, missing value counts, outlier "
                "flags, and Pearson correlations."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "csv_text": {
                        "type":        "string",
                        "description": "Raw CSV content as a string.",
                    },
                },
                "required": ["csv_text"],
            },
        },
    ]
}


# =========================================================
# Quick smoke test
# =========================================================

if __name__ == "__main__":
    import pathlib
    csv_path = pathlib.Path("data.csv")
    if csv_path.exists():
        csv_text = csv_path.read_text()
        print("=== EDA Tool ===")
        eda_result = json.loads(eda_tool(csv_text))
        print(f"  Shape: {eda_result['shape']}")
        print(f"  Columns: {list(eda_result['column_types'].keys())}")

        print("\n=== AutoML Tool (classification) ===")
        result = json.loads(automl_tool(csv_text, target="churned", task="classification", verbose=True))
        print(result.get("agent_summary", "No summary."))
    else:
        print("data.csv not found — skipping smoke test.")
