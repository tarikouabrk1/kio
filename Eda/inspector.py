import numpy as np
from math_utils.statistics import (
    mean, median, mode, std, count, quantile,
    col_min, col_max, skewness, kurtosis, pearson_correlation,
)


class EDAInspector:
    """
    White-box Automated EDA module.

    Usage
    -----
    inspector = EDAInspector(df)
    report    = inspector.run()          # full JSON-serialisable report
    inspector.print_report()             # pretty console output
    """

    def __init__(self, df):
        """
        Parameters
        ----------
        df : DataFrame
            A GlassBox DataFrame (must expose .dtypes, .get_features(),
            .get_numerical(), .get_object()).
        """
        self.df = df
        self._report = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> dict:
        """Run all EDA checks and return a structured report dict."""
        self._report = {
            "shape": self._shape(),
            "column_types": self._auto_type(),
            "profiles": self._profile_all(),
            "outliers": self._detect_outliers_all(),
            "correlations": self._correlation_matrix(),
            "missing_summary": self._missing_summary(),
        }
        return self._report

    def print_report(self):
        """Pretty-print the EDA report to stdout."""
        if self._report is None:
            self.run()
        r = self._report
        n_rows, n_cols = r["shape"]
        print("\n" + "=" * 60)
        print(f"  GlassBox EDA Inspector  —  {n_rows} rows × {n_cols} cols")
        print("=" * 60)

        print("\n── Column types ──────────────────────────────────────────")
        for col, dtype in r["column_types"].items():
            print(f"  {col:<25} {dtype}")

        print("\n── Missing values ────────────────────────────────────────")
        for col, info in r["missing_summary"].items():
            if info["missing_count"] > 0:
                print(f"  {col:<25} {info['missing_count']} missing  "
                      f"({info['missing_pct']:.1f}%)")

        print("\n── Numerical profiles ────────────────────────────────────")
        for col, p in r["profiles"].items():
            if p["dtype"] not in ("int", "float"):
                continue
            print(f"\n  {col}")
            for k, v in p.items():
                if k == "dtype":
                    continue
                val = f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v)
                print(f"    {k:<18} {val}")

        print("\n── Categorical profiles ──────────────────────────────────")
        for col, p in r["profiles"].items():
            if p["dtype"] != "str":
                continue
            print(f"\n  {col}")
            print(f"    unique values      {p['unique_count']}")
            print(f"    mode               {p['mode']}")
            print(f"    missing            {p['missing_count']}")

        print("\n── Outliers (IQR method) ─────────────────────────────────")
        for col, info in r["outliers"].items():
            if info["n_outliers"] > 0:
                print(f"  {col:<25} {info['n_outliers']} outliers  "
                      f"IQR [{info['lower_fence']:.2f}, {info['upper_fence']:.2f}]")

        print("\n── Pearson correlation (top pairs |r| ≥ 0.5) ────────────")
        pairs = r["correlations"].get("top_pairs", [])
        if pairs:
            for p in pairs:
                print(f"  {p['col_a']:<15} × {p['col_b']:<15}  r = {p['r']:.4f}")
        else:
            print("  No pairs with |r| ≥ 0.5 found.")

        print()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _shape(self):
        return self.df.format()          # (n_rows, n_cols)

    def _auto_type(self) -> dict:
        """Return the detected dtype for every column."""
        return dict(self.df.dtypes)

    def _missing_summary(self) -> dict:
        n_rows, _ = self.df.format()
        result = {}
        for col in self.df.get_features():
            col_data = self.df[col]
            try:
                arr = np.array(col_data, dtype=np.float64)
                n_missing = int(np.isnan(arr).sum())
            except (ValueError, TypeError):
                arr = np.array(col_data, dtype=object)
                n_missing = int(sum(
                    1 for v in arr
                    if v is None or (isinstance(v, float) and np.isnan(v))
                ))
            result[col] = {
                "missing_count": n_missing,
                "missing_pct": round(100.0 * n_missing / n_rows, 2) if n_rows else 0.0,
            }
        return result

    def _profile_numerical(self, col_name: str) -> dict:
        arr = np.array(self.df[col_name], dtype=np.float64)
        valid = arr[~np.isnan(arr)]
        if valid.size == 0:
            return {"dtype": self.df.get_type(col_name), "missing_count": int(arr.size)}
        return {
            "dtype":        self.df.get_type(col_name),
            "count":        int(count(arr)),
            "missing_count": int(np.isnan(arr).sum()),
            "mean":         float(mean(valid)),
            "median":       float(median(valid)),
            "mode":         float(mode(valid)),
            "std":          float(std(valid)),
            "min":          float(col_min(valid)),
            "max":          float(col_max(valid)),
            "q25":          float(quantile(valid, 0.25)),
            "q75":          float(quantile(valid, 0.75)),
            "skewness":     float(skewness(valid)),
            "kurtosis":     float(kurtosis(valid)),
        }

    def _profile_categorical(self, col_name: str) -> dict:
        arr = np.array(self.df[col_name], dtype=object)
        missing_mask = np.array([
            v is None or (isinstance(v, float) and np.isnan(v))
            for v in arr
        ])
        valid = arr[~missing_mask]
        unique_vals, counts = np.unique(valid, return_counts=True) if valid.size else (np.array([]), np.array([]))
        top = {}
        if unique_vals.size:
            order = np.argsort(-counts)
            for i in order[:5]:
                top[str(unique_vals[i])] = int(counts[i])
        return {
            "dtype":         "str",
            "missing_count": int(missing_mask.sum()),
            "unique_count":  int(unique_vals.size),
            "mode":          str(unique_vals[np.argmax(counts)]) if unique_vals.size else None,
            "top_values":    top,
        }

    def _profile_all(self) -> dict:
        profiles = {}
        for col, dtype in self.df.dtypes.items():
            if dtype in ("int","bool" "float"):
                profiles[col] = self._profile_numerical(col)
            else:
                profiles[col] = self._profile_categorical(col)
        return profiles

    def _detect_outliers_col(self, col_name: str) -> dict:
        arr = np.array(self.df[col_name], dtype=np.float64)
        valid = arr[~np.isnan(arr)]
        if valid.size < 4:
            return {"n_outliers": 0, "lower_fence": None, "upper_fence": None, "outlier_indices": []}
        q1 = float(quantile(valid, 0.25))
        q3 = float(quantile(valid, 0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (arr < lower) | (arr > upper)
        indices = [int(i) for i in np.where(mask)[0]]
        return {
            "n_outliers":    len(indices),
            "lower_fence":   round(lower, 4),
            "upper_fence":   round(upper, 4),
            "outlier_indices": indices,
        }

    def _detect_outliers_all(self) -> dict:
        result = {}
        for col, dtype in self.df.dtypes.items():
            if dtype in ("int", "float"):
                result[col] = self._detect_outliers_col(col)
        return result

    def _correlation_matrix(self) -> dict:
        num_df = self.df.get_numerical()
        cols = num_df.get_features()
        if len(cols) < 2:
            return {"columns": cols, "matrix": [], "top_pairs": []}

        # Build a clean matrix (drop rows with any NaN)
        rows = []
        for col in cols:
            rows.append(np.array(num_df[col], dtype=np.float64))
        X = np.column_stack(rows)
        valid_mask = ~np.isnan(X).any(axis=1)
        X_clean = X[valid_mask]

        if X_clean.shape[0] < 2:
            return {"columns": cols, "matrix": [], "top_pairs": []}

        matrix = pearson_correlation(X_clean)

        # Serialise matrix
        matrix_list = [[round(float(v), 4) for v in row] for row in matrix]

        # Extract top correlated pairs (|r| >= 0.5, upper triangle only)
        top_pairs = []
        n = len(cols)
        for i in range(n):
            for j in range(i + 1, n):
                r = matrix[i, j]
                if abs(r) >= 0.5:
                    top_pairs.append({
                        "col_a": cols[i],
                        "col_b": cols[j],
                        "r":     round(float(r), 4),
                    })
        top_pairs.sort(key=lambda x: abs(x["r"]), reverse=True)

        return {
            "columns":   cols,
            "matrix":    matrix_list,
            "top_pairs": top_pairs,
        }


# ------------------------------------------------------------------
# Quick self-test
# ------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from dataframes.DataFrame import DataFrame

    df = DataFrame.load_csv("data.csv")
    inspector = EDAInspector(df)
    inspector.run()
    inspector.print_report()