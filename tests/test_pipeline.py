import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataframes.DataFrame import DataFrame
from pipeline.autofit import AutoFit


class AutoFitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = DataFrame.load_csv(ROOT / "data.csv")

    def test_classification_report_includes_confusion_matrix(self):
        report = AutoFit(
            target="churned",
            task="classification",
            cv=3,
            verbose=False,
        ).fit(self.df)

        self.assertIn("confusion_matrix", report["eval_metrics"])
        self.assertIsInstance(report["eval_metrics"]["confusion_matrix"], list)

    def test_predict_accepts_feature_only_dataframe(self):
        pipeline = AutoFit(
            target="churned",
            task="classification",
            cv=3,
            verbose=False,
        )
        pipeline.fit(self.df)

        features = [name for name in self.df.get_features() if name != "churned"]
        feature_df = DataFrame(
            self.df[features],
            dtypes={name: self.df.dtypes[name] for name in features},
        )
        preds = pipeline.predict(feature_df)

        self.assertEqual(len(preds), len(feature_df))

    def test_random_search_path_runs(self):
        report = AutoFit(
            target="churned",
            task="classification",
            cv=3,
            search_strategy="random",
            random_iter=4,
            verbose=False,
        ).fit(self.df)

        self.assertTrue(report["search_results"])
        self.assertIsNotNone(report["best_model"])

    def test_regression_path_runs(self):
        report = AutoFit(
            target="lifetime_value",
            task="regression",
            cv=3,
            verbose=False,
        ).fit(self.df)

        self.assertIn("R2", report["eval_metrics"])


if __name__ == "__main__":
    unittest.main()

