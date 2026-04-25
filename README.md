# GlassBoxAI

GlassBoxAI is a scratch-built AutoML library built mostly with NumPy. It includes:

- Automated EDA for profiling, missing values, outliers, and correlations
- Preprocessing utilities for imputation, scaling, and encoding
- Pure-Python/NumPy models for classification and regression
- Hyperparameter search with grid search, random search, and K-fold CV support
- An agent-facing tool wrapper that returns JSON reports

## Project Layout

- `dataframes/`: custom dataframe loader and typing
- `Eda/`: automated inspection and profiling
- `preprocessing/`: imputers, encoders, and scalers
- `models/`: linear, logistic, KNN, naive Bayes, tree, and forest models
- `optimization/`: grid search, random search, and K-fold logic
- `evaluation/`: classification and regression metrics
- `pipeline/`: end-to-end AutoFit pipeline
- `agent/`: tool wrapper for agent integration

## Example

```python
from dataframes.DataFrame import DataFrame
from pipeline.autofit import AutoFit

df = DataFrame.load_csv("data.csv")

pipeline = AutoFit(
    target="churned",
    task="classification",
    cv=3,
    search_strategy="grid",
    verbose=False,
)

report = pipeline.fit(df)
predictions = pipeline.predict(df[[c for c in df.get_features() if c != "churned"]])
```

## Notes

- `AutoFit.predict()` now works on feature-only dataframes.
- Classification reports include a confusion matrix in the returned JSON.
- The agent wrapper reads CSV content from memory instead of using temp files.
