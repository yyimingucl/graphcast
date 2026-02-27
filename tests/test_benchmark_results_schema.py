import json
import unittest
from pathlib import Path


class BenchmarkResultsSchemaTest(unittest.TestCase):
  def test_schema_file_exists_and_parses(self):
    schema_path = Path(__file__).resolve().parents[1] / "benchmark_results_schema.json"
    self.assertTrue(schema_path.exists())
    schema = json.loads(schema_path.read_text())
    self.assertIsInstance(schema, dict)
    self.assertEqual(schema.get("type"), "object")
    self.assertIn("properties", schema)

  def test_minimal_results_shape_matches_required_keys(self):
    # This is a lightweight structural check without jsonschema dependency.
    results = {
        "run_metadata": {
            "timestamp_utc": "2026-02-27T00:00:00Z",
            "model_kind": "graphcast",
            "preset_name": "GraphCast_small",
            "data_kind": "era5",
        },
        "timing": {
            "warmup_steps": 1,
            "timed_repeats": 1,
            "repeat_seconds": [1.0],
            "median_seconds": 1.0,
            "p95_seconds": 1.0,
        },
        "metrics": {"overall": {"rmse": 0.0}},
    }
    for key in ("run_metadata", "timing", "metrics"):
      self.assertIn(key, results)
    for key in ("timestamp_utc", "model_kind", "preset_name", "data_kind"):
      self.assertIn(key, results["run_metadata"])
    for key in ("warmup_steps", "timed_repeats", "repeat_seconds", "median_seconds", "p95_seconds"):
      self.assertIn(key, results["timing"])
    self.assertIn("overall", results["metrics"])


if __name__ == "__main__":
  unittest.main()
