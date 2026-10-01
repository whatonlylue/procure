"""Coverage for the offline retrieval benchmark harness (app.bench).

Stdlib unittest only (no new harness). Run from the project root:

    .venv/bin/python -m unittest discover -s tests
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest

from app.bench import datasets, metrics
from app.bench.runner import BenchmarkRunner, ingest_corpus, remap_queries, run_benchmark
from app.config import Settings
from app.service import RAGService


def _settings(tmp: str) -> Settings:
    s = Settings()
    s.data_dir = tmp
    s.embeddings = "hash"
    return s


def _service(tmp: str) -> RAGService:
    return RAGService(_settings(tmp))


class MetricsTest(unittest.TestCase):
    def test_recall_precision_hit(self) -> None:
        retrieved = ["a", "b", "c"]
        relevant = {"b", "d"}
        self.assertAlmostEqual(metrics.recall_at_k(retrieved, relevant, 2), 0.5)
        self.assertAlmostEqual(metrics.precision_at_k(retrieved, relevant, 2), 0.5)
        self.assertEqual(metrics.hit_rate_at_k(retrieved, relevant, 2), 1.0)
        self.assertEqual(metrics.hit_rate_at_k(["a"], relevant, 1), 0.0)

    def test_empty_relevant_scores_zero(self) -> None:
        self.assertEqual(metrics.recall_at_k(["a"], set(), 5), 0.0)
        self.assertEqual(metrics.ndcg_at_k(["a"], set(), 5), 0.0)
        self.assertEqual(metrics.hit_rate_at_k(["a"], set(), 5), 0.0)

    def test_mrr_first_hit_rank(self) -> None:
        pairs = [(["x", "gold"], {"gold"}), (["nope"], {"gold"})]
        self.assertAlmostEqual(metrics.mrr(pairs), 0.25)

    def test_ndcg_perfect_and_partial(self) -> None:
        self.assertAlmostEqual(
            metrics.ndcg_at_k(["gold", "other"], {"gold"}, 2), 1.0)
        partial = metrics.ndcg_at_k(["other", "gold"], {"gold"}, 2)
        self.assertGreater(partial, 0.0)
        self.assertLess(partial, 1.0)

    def test_latency_stats(self) -> None:
        stats = metrics.latency_stats([0.1, 0.2, 0.3, 0.4])
        self.assertEqual(stats["count"], 4)
        self.assertGreaterEqual(stats["p95_s"], stats["p50_s"])
        self.assertGreaterEqual(stats["p50_s"], stats["min_s"])
        empty = metrics.latency_stats([])
        self.assertEqual(empty["count"], 0)


class DatasetsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _write(self, name: str, lines: list[dict]) -> str:
        path = os.path.join(self.tmp.name, name)
        with open(path, "w", encoding="utf-8") as f:
            for row in lines:
                f.write(json.dumps(row) + "\n")
        return path

    def test_round_trip_sample(self) -> None:
        paths = datasets.write_sample_dataset(
            os.path.join(self.tmp.name, "sample"))
        corpus = datasets.load_corpus_jsonl(paths["corpus"])
        queries = datasets.load_queries_jsonl(paths["queries"])
        self.assertEqual(len(corpus), 3)
        self.assertEqual(len(queries), 4)
        corpus_ids = {d["doc_id"] for d in corpus}
        for q in queries:
            self.assertTrue(set(q["relevant_doc_ids"]) <= corpus_ids)

    def test_rejects_query_without_gold(self) -> None:
        path = self._write("q.jsonl", [{"query": "orphan"}])
        with self.assertRaises(ValueError):
            datasets.load_queries_jsonl(path)

    def test_rejects_duplicate_doc_ids(self) -> None:
        path = self._write("c.jsonl", [
            {"doc_id": "dup", "text": "hello world " * 30},
            {"doc_id": "dup", "text": "other words here " * 30},
        ])
        with self.assertRaises(ValueError):
            datasets.load_corpus_jsonl(path)

    def test_beir_dir(self) -> None:
        bdir = os.path.join(self.tmp.name, "beir")
        os.makedirs(bdir)
        with open(os.path.join(bdir, "corpus.jsonl"), "w",
                  encoding="utf-8") as f:
            f.write(json.dumps({"_id": "d1", "title": "T",
                                "text": "lighthouse beacon relamped"}) + "\n")
        with open(os.path.join(bdir, "queries.jsonl"), "w",
                  encoding="utf-8") as f:
            f.write(json.dumps({"_id": "q1",
                                "text": "when relamped beacon?"}) + "\n")
        with open(os.path.join(bdir, "qrels.tsv"), "w",
                  encoding="utf-8") as f:
            f.write("q1\td1\t1\n")
        corpus, queries = datasets.load_beir_dir(bdir)
        self.assertEqual(corpus[0]["doc_id"], "d1")
        self.assertEqual(queries[0]["relevant_doc_ids"], ["d1"])


class RunnerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.svc = _service(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_sample_dataset_recalls_gold(self) -> None:
        report = run_benchmark(self.svc, datasets.SAMPLE_CORPUS,
                               datasets.SAMPLE_QUERIES, (5, 10))
        self.assertEqual(report.summary["num_queries"], 4)
        self.assertAlmostEqual(report.summary["@5"]["recall"], 1.0)
        self.assertAlmostEqual(report.summary["@10"]["recall"], 1.0)
        self.assertEqual(report.summary["zero_result_rate"], 0.0)
        self.assertEqual(report.summary["latency"]["count"], 4)
        self.assertIn("search_mode", report.config)
        payload = report.to_dict()
        self.assertEqual(len(payload["queries"]), 4)
        self.assertIn("recall@k", report.to_markdown())

    def test_chunk_level_recall(self) -> None:
        mapping = ingest_corpus(self.svc, datasets.SAMPLE_CORPUS[:1])
        lib_id = mapping["lighthouse"]
        runner = BenchmarkRunner(self.svc, (5,))
        res = runner.run_query("north beacon relamped", [],
                               [f"{lib_id}:0"])
        self.assertIn(f"{lib_id}:0", res.retrieved_ids)
        self.assertEqual(res.scores["@5"]["recall"], 1.0)

    def test_remap_drops_unknown_docs(self) -> None:
        out = remap_queries(
            [{"query": "q", "relevant_doc_ids": ["ghost"],
              "relevant_chunk_ids": []}],
            {"lighthouse": "abc123"})
        self.assertEqual(out, [])

    def test_doc_level_dedupes_repeat_chunks(self) -> None:
        """Repeated doc ids (one doc, many chunks) must not push nDCG > 1."""

        class _FakeService:
            settings = _settings(tempfile.mkdtemp())

            def search(self, query: str, top_k: int) -> list[dict]:
                del query, top_k
                return [{"chunk_id": f"docA:{i}", "doc_id": "docA"}
                        for i in range(6)]

        runner = BenchmarkRunner(_FakeService(), (5, 10))
        res = runner.run_query("q", ["docA"], [])
        self.assertEqual(res.retrieved_ids, ["docA"])
        self.assertLessEqual(res.scores["@5"]["ndcg"], 1.0)
        self.assertEqual(res.scores["@5"]["recall"], 1.0)

    def test_failed_ingest_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            ingest_corpus(self.svc, [{
                "doc_id": "empty", "filename": "empty.md", "text": "",
                "tags": [], "doc_type": "document",
            }])

    def test_short_corpus_docs_ingest(self) -> None:
        """BEIR stubs shorter than the chunk floor must not abort the run."""
        corpus = [
            {"doc_id": "stub", "filename": "stub.md",
             "text": "List of cryptographers\n\nList of cryptographers.",
             "tags": [], "doc_type": "document"},
            {"doc_id": "lighthouse", "filename": "lighthouse.md",
             "text": datasets.SAMPLE_CORPUS[0]["text"],
             "tags": [], "doc_type": "document"},
        ]
        mapping = ingest_corpus(self.svc, corpus)
        self.assertEqual(set(mapping), {"stub", "lighthouse"})
        report = run_benchmark(
            self.svc, corpus,
            [{"query": "cryptographers list",
              "relevant_doc_ids": ["stub"], "relevant_chunk_ids": []}],
            (5,))
        self.assertEqual(report.summary["num_queries"], 1)
        self.assertEqual(report.summary["@5"]["recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
