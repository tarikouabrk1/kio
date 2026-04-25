# GlassBoxAI

GlassBoxAI is a scratch-built AutoML project that focuses on transparency. Instead of hiding the workflow behind heavy ML frameworks, it implements the core pipeline directly in Python and NumPy: data loading, EDA, preprocessing, model training, hyperparameter search, evaluation, and agent-friendly reporting.

The goal of the project is to behave like a small white-box AutoML engine that can inspect a CSV dataset, prepare it, try multiple models, evaluate them, and return a structured report.

## What The Project Does

GlassBoxAI currently supports:

- Automated EDA for dataset shape, missing values, numerical profiling, categorical profiling, outlier detection, and correlation analysis
- Preprocessing tools for imputation, scaling, one-hot encoding, and label encoding
- Multiple scratch-built models for both classification and regression
- Hyperparameter search with grid search, random search, and K-fold cross-validation support
- A high-level `AutoFit` pipeline that ties the workflow together
- An agent wrapper that returns JSON reports for tool-based integration

## Main Features

### 1. Custom Data Handling

The project uses a custom `DataFrame` class built on top of NumPy arrays. It supports:

- CSV loading
- basic type inference
- numerical / categorical column access
- feature listing
- simple metadata tracking through `dtypes`

It now also detects boolean-style columns like `0/1` as `bool`.

### 2. Automated EDA

The EDA inspector provides:

- dataset shape
- detected column types
- missing value summaries
- numerical statistics
- categorical summaries
- IQR-based outlier detection
- Pearson correlation analysis

This makes it easier to understand the data before training.

### 3. Preprocessing

The preprocessing layer includes:

- `SimpleImputer`
- `MeanImputer`
- `MedianImputer`
- `ModeImputer`
- `StandardScaler`
- `MinMaxScaler`
- `LabelEncoder`
- `OneHotEncoder`

These are used directly inside the AutoML flow.

### 4. Model Zoo

The project currently includes implementations for:

- Linear Regression
- Logistic Regression
- KNN Classifier / Regressor
- Naive Bayes
- Decision Tree Classifier / Regressor
- Random Forest Classifier / Regressor

### 5. Optimization

The optimization layer includes:

- Grid Search
- Random Search
- K-Fold splitting

`AutoFit` can now use either `grid` or `random` search, and it passes through the configured `cv` value during model search.

### 6. Evaluation

Classification outputs include:

- accuracy
- precision
- recall
- F1 score
- confusion matrix

Regression outputs include:

- MAE
- MSE
- RMSE
- R2

### 7. Agent Integration

The `agent/tools.py` module exposes:

- `automl_tool(...)`
- `eda_tool(...)`

These return JSON so the project can be used as part of an agent workflow. The CSV loader in the agent path now works directly from memory rather than writing temporary files first.

## Project Structure

```text
GlassBoxAI/
├── agent/           # Agent-facing tool wrapper and schema
├── dataframes/      # Custom DataFrame loader and typing helpers
├── Eda/             # Automated EDA inspector
├── evaluation/      # Classification and regression metrics
├── math_utils/      # Statistics and distance helpers
├── models/          # ML model implementations
├── optimization/    # Grid search, random search, K-fold
├── pipeline/        # End-to-end AutoFit pipeline
├── preprocessing/   # Imputers, scalers, encoders
├── tests/           # Automated unittest suite
├── data.csv         # Sample dataset
└── test.ipynb       # Notebook-based exploration / smoke test
```

## Quick Start

Run from the project root:

```powershell
cd C:\Users\PC\Desktop\dsd\GlassBoxAI
```

### Classification Example

```powershell
@'
from dataframes.DataFrame import DataFrame
from pipeline.autofit import AutoFit

df = DataFrame.load_csv("data.csv")
report = AutoFit(
    target="churned",
    task="classification",
    cv=3,
    search_strategy="grid",
    verbose=True,
).fit(df)

print(report["best_model"])
print(report["eval_metrics"])
'@ | python -
```

### Regression Example

```powershell
@'
from dataframes.DataFrame import DataFrame
from pipeline.autofit import AutoFit

df = DataFrame.load_csv("data.csv")
report = AutoFit(
    target="lifetime_value",
    task="regression",
    cv=3,
    search_strategy="grid",
    verbose=True,
).fit(df)

print(report["best_model"])
print(report["eval_metrics"])
'@ | python -
```

### Prediction On Feature-Only Data

```powershell
@'
from dataframes.DataFrame import DataFrame
from pipeline.autofit import AutoFit

df = DataFrame.load_csv("data.csv")
pipe = AutoFit(target="churned", task="classification", cv=3, verbose=False)
pipe.fit(df)

features = [c for c in df.get_features() if c != "churned"]
new_df = DataFrame(df[features], dtypes={c: df.dtypes[c] for c in features})
print(pipe.predict(new_df)[:5])
'@ | python -
```

## Using The Agent Wrapper

```powershell
@'
import json
from pathlib import Path
from agent.tools import automl_tool

csv_text = Path("data.csv").read_text()
result = json.loads(
    automl_tool(
        csv_text,
        target="churned",
        task="classification",
        cv=3,
        search_strategy="grid",
    )
)

print(result["best_model"])
print(result["agent_summary"])
'@ | python -
```

## How To Test The Project

### Run The Full Automated Test Suite

```powershell
python -m unittest discover -s tests -v
```

This currently checks:

- dataframe loading and type inference
- feature-only prediction flow
- classification pipeline behavior
- regression pipeline behavior
- random-search execution path
- agent JSON tool outputs

### Run Individual Test Files

```powershell
python -m unittest tests.test_dataframe -v
python -m unittest tests.test_pipeline -v
python -m unittest tests.test_agent -v
```

### Recommended Testing Order

1. Run the full automated suite.
2. Run one manual `AutoFit` classification example.
3. Run one feature-only prediction example.
4. Run the agent wrapper example.

This gives both automated verification and a real end-to-end workflow check.

## Public API Surface

The simplest imports are:

```python
from dataframes import DataFrame
from pipeline import AutoFit
from agent import automl_tool, eda_tool
```

You can also import from the project root package:

```python
from GlassBoxAI import DataFrame, AutoFit, automl_tool, eda_tool
```

## Current Status

This project is in a strong MVP state.

What is working well:

- core AutoML workflow
- custom preprocessing and evaluation
- model search
- agent-facing JSON output
- automated tests for the main paths

What is still rough:

- some older console strings still have encoding artifacts
- naming/style consistency can be improved further
- packaging is not yet set up like a published PyPI library
- the test suite covers important flows, but not every edge case

## Limitations

GlassBoxAI is a solid scratch-built project, but it is still a project library rather than a production ML framework. A few things to keep in mind:

- it is not benchmarked broadly against established libraries yet
- it does not have industrial-scale dataset support
- it does not yet have a formal packaging / distribution setup
- some algorithms and interfaces would benefit from more edge-case hardening

## Why This Project Is Interesting

The value of GlassBoxAI is not just that it trains a model. The interesting part is that the whole flow is inspectable. You can read how the data is typed, how missing values are handled, how features are encoded, how models are trained, and how the final choice is made.

That makes it useful as:

- a learning project
- a portfolio project
- a base for an explainable AutoML tool
- a starting point for agent-based ML workflows

## Future Improvements

Good next steps would be:

- clean up the remaining output text / encoding artifacts
- add more datasets and broader tests
- improve package ergonomics and installation setup
- document each model and optimizer more deeply
- add more robust error handling and edge-case validation
- benchmark against scikit-learn baselines

## Summary

GlassBoxAI is a transparent, modular AutoML project with real end-to-end behavior. It now has a clearer structure, working pipeline paths, agent integration, and an automated test suite. The foundation is solid, and the next stage is mostly maturity and polish rather than rescue work.
