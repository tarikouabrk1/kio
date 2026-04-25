import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.tools import automl_tool, eda_tool


class AgentToolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.csv_text = (ROOT / "data.csv").read_text()

    def test_eda_tool_returns_expected_sections(self):
        payload = json.loads(eda_tool(self.csv_text))

        self.assertIn("shape", payload)
        self.assertIn("profiles", payload)
        self.assertIn("correlations", payload)

    def test_automl_tool_returns_json_report(self):
        payload = json.loads(
            automl_tool(
                self.csv_text,
                target="churned",
                task="classification",
                cv=3,
                search_strategy="grid",
            )
        )

        self.assertEqual(payload["target"], "churned")
        self.assertIn("agent_summary", payload)
        self.assertIn("confusion_matrix", payload["eval_metrics"])


if __name__ == "__main__":
    unittest.main()
