"""Regression guard for the SciFact retrieval baseline.

Compares the saved potion-embedder baseline
(``data/benchmarks/scifact-report-potion.json``) against hardcoded quality
floors (2026-10-04 numbers minus epsilon) and, when the hash-era report is
present, against the old report too. Future optimizations regenerate the
baseline file via the bench harness; this test fails if the new numbers
drop below the floor.

Pure JSON comparison: no model download, no ingest, runs in milliseconds.
Skips when the baseline file is absent (benchmark data lives under the
gitignored ``data/`` dir, so a fresh clone has nothing to check yet).
"""
from __future__ import annotations

import json
import os
import unittest

# Floors sit just below the 2026-10-04 baseline (hybrid, rerank on):
# recall@5 0.7519 · nDCG@5 0.6673 · recall@10 0.7828 · MRR 0.6514.
FLOORS = {
    "recall@5": 0.74,
    "ndcg@5": 0.65,
    "recall@10": 0.77,
    "mrr": 0.63,
}

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEW_PATH = os.path.join(ROOT, "data", "benchmarks",
                         "scifact-report-potion.json")
OLD_PATH = os.path.join(ROOT, "data", "benchmarks", "scifact-report.json")


def _load(path: str):
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class BaselineQualityTest(unittest.TestCase):
    def test_baseline_clears_quality_floors(self) -> None:
        report = _load(NEW_PATH)
        if report is None:
            self.skipTest(f"no baseline yet: {NEW_PATH}")
        summary = report["summary"]
        self.assertEqual(summary["num_queries"], 300)
        self.assertEqual(summary["num_docs"], 5183)
        self.assertEqual(report["ks"], [5, 10])
        self.assertEqual(summary["zero_result_rate"], 0.0)
        observed = {
            "recall@5": summary["@5"]["recall"],
            "ndcg@5": summary["@5"]["ndcg"],
            "recall@10": summary["@10"]["recall"],
            "mrr": summary["mrr"],
        }
        for name, floor in FLOORS.items():
            self.assertGreaterEqual(
                observed[name], floor,
                f"{name} regressed: {observed[name]} < floor {floor}")

    def test_baseline_config_is_default_pipeline(self) -> None:
        report = _load(NEW_PATH)
        if report is None:
            self.skipTest(f"no baseline yet: {NEW_PATH}")
        config = report["config"]
        self.assertEqual(config["search_mode"], "hybrid")
        self.assertEqual(config["rerank"], "on")
        self.assertIn("potion", config["embeddings"])

    def test_baseline_no_worse_than_hash_era(self) -> None:
        new = _load(NEW_PATH)
        old = _load(OLD_PATH)
        if new is None:
            self.skipTest(f"no baseline yet: {NEW_PATH}")
        if old is None:
            self.skipTest(f"no old report: {OLD_PATH}")
        for metric in ("recall", "ndcg"):
            self.assertGreaterEqual(
                new["summary"]["@5"][metric], old["summary"]["@5"][metric],
                f"@5 {metric} fell below the hash-era report")
        self.assertGreaterEqual(new["summary"]["mrr"], old["summary"]["mrr"],
                                "MRR fell below the hash-era report")


if __name__ == "__main__":
    unittest.main()
