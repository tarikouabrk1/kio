import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataframes.DataFrame import DataFrame


class DataFrameTests(unittest.TestCase):
    def test_load_csv_infers_bool_and_numeric_types(self):
        df = DataFrame.load_csv(ROOT / "data.csv")

        self.assertEqual(df.dtypes["churned"], "bool")
        self.assertEqual(df.dtypes["age"], "int")
        self.assertIn("monthly_spend", df.get_numerical().get_features())

    def test_feature_subset_preserves_dataframe_shape(self):
        df = DataFrame.load_csv(ROOT / "data.csv")
        features = [name for name in df.get_features() if name != "churned"]
        feature_df = DataFrame(df[features], dtypes={name: df.dtypes[name] for name in features})

        self.assertEqual(feature_df.format()[0], df.format()[0])
        self.assertNotIn("churned", feature_df.get_features())


if __name__ == "__main__":
    unittest.main()

