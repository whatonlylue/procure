"""Coverage for BEIR/MTEB dataset pulling (app.bench.sources).

Stdlib unittest only (no new harness). Network is stubbed with a fake
fetcher or a local HTTP server — nothing here hits the internet.
Run from the project root:

    .venv/bin/python -m unittest discover -s tests
"""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
import urllib.error
from unittest import mock

from app.bench import datasets, sources
from app.cli import main as cli_main

NATIVE_CORPUS = [
    {"_id": "d1", "title": "Beacon", "text": "The north beacon was relamped in March."},
    {"_id": "d2", "title": "Tides", "text": "High water at dawn near the east pier."},
    {"_id": "d3", "title": "", "text": "Sourdough proofs for six hours."},
]

NATIVE_QUERIES = [
    {"_id": "q1", "text": "when was the beacon relamped?"},
    {"_id": "q2", "text": "when is high water?"},
    {"_id": "q3", "text": "orphan query with no qrels"},
]

NATIVE_QRELS = "query-id\tcorpus-id\tscore\nq1\td1\t1\nq2\td2\t1\nq9\td9\t0\n"


def _fake_fetcher(url: str, dest: str) -> str:
    """Serve fixture natives instead of the network."""
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    if dest.endswith("corpus.jsonl"):
        rows = NATIVE_CORPUS
    elif dest.endswith("queries.jsonl"):
        rows = NATIVE_QUERIES
    elif dest.endswith("qrels.tsv"):
        with open(dest, "w", encoding="utf-8") as f:
            f.write(NATIVE_QRELS)
        return dest
    else:
        raise AssertionError(f"unexpected pull target: {url} -> {dest}")
    with open(dest, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return dest


class RegistryTest(unittest.TestCase):
    def test_keys_and_repos(self) -> None:
        rows = sources.list_datasets()
        self.assertGreater(len(rows), 5)
        for r in rows:
            self.assertEqual(r["dataset"], r["dataset"].lower())
            self.assertIn("/", r["mteb"])
            self.assertTrue(r["beir"].startswith("BeIR/"))
            self.assertTrue(r["beir_qrels"].endswith("-qrels"))

    def test_resolve_known_and_explicit(self) -> None:
        self.assertEqual(sources.resolve_repo("scifact", "mteb"), "mteb/scifact")
        self.assertEqual(sources.resolve_repo("scifact", "beir"), "BeIR/scifact")
        self.assertEqual(sources.resolve_repo("mteb/custom-set", "mteb"),
                         "mteb/custom-set")

    def test_resolve_rejects_unknown(self) -> None:
        with self.assertRaises(ValueError):
            sources.resolve_repo("nope", "mteb")
        with self.assertRaises(ValueError):
            sources.resolve_repo("scifact", "arxiv")
        with self.assertRaises(ValueError):
            sources.resolve_repo("BeIR/scifact", "beir")
        with self.assertRaises(ValueError):
            sources.pull("scifact", "mteb", split="val")


class PullMtebTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dest = os.path.join(self.tmp.name, "scifact")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_full_pull_converts(self) -> None:
        out = sources.pull("scifact", "mteb", "test", None, self.dest,
                           fetcher=_fake_fetcher)
        self.assertEqual(out["num_docs"], 3)
        self.assertEqual(out["num_queries"], 2)  # q3 has no qrels
        corpus = datasets.load_corpus_jsonl(out["corpus"])
        queries = datasets.load_queries_jsonl(out["queries"])
        self.assertEqual(len(corpus), 3)
        self.assertIn("Beacon", corpus[0]["text"])
        self.assertEqual(queries[0]["relevant_doc_ids"], ["d1"])
        with open(out["manifest"], encoding="utf-8") as f:
            manifest = json.load(f)
        self.assertEqual(manifest["source"], "mteb")
        self.assertEqual(manifest["num_docs"], 3)

    def test_max_queries_keeps_full_corpus(self) -> None:
        out = sources.pull("scifact", "mteb", "test", 1, self.dest,
                           fetcher=_fake_fetcher)
        self.assertEqual(out["num_queries"], 1)
        self.assertEqual(out["num_docs"], 3)
        self.assertEqual(
            len(datasets.load_corpus_jsonl(out["corpus"])), 3)

    def test_missing_split_raises(self) -> None:
        def _no_qrels(url: str, dest: str) -> str:
            if "qrels" in url:
                raise RuntimeError("download failed (404)")
            return _fake_fetcher(url, dest)

        with self.assertRaises(RuntimeError):
            sources.pull("scifact", "mteb", "dev", None, self.dest,
                         fetcher=_no_qrels)

    def test_empty_corpus_raises(self) -> None:
        def _empty(url: str, dest: str) -> str:
            if dest.endswith("corpus.jsonl"):
                with open(dest, "w", encoding="utf-8") as f:
                    f.write('{"_id": "", "text": "  "}\n')
                return dest
            return _fake_fetcher(url, dest)

        with self.assertRaises(RuntimeError):
            sources.pull("scifact", "mteb", "test", None, self.dest,
                         fetcher=_empty)


class QrelsTest(unittest.TestCase):
    def test_header_and_zero_scores_skipped(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".tsv",
                                         delete=False) as f:
            f.write(NATIVE_QRELS)
            path = f.name
        try:
            rel = sources._read_qrels_tsv(path)
        finally:
            os.remove(path)
        self.assertEqual(rel, {"q1": {"d1"}, "q2": {"d2"}})


class _FakeResponse:
    """Minimal urlopen response (the sandbox forbids listening sockets,
    so stream logic is tested with a stub instead of a local server)."""

    def __init__(self, payload: bytes):
        self._buf = io.BytesIO(payload)
        self._len = str(len(payload))

    def getheader(self, name: str):
        return self._len if name == "Content-Length" else None

    def read(self, n: int = -1) -> bytes:
        return self._buf.read(n)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class DownloadFileTest(unittest.TestCase):
    def test_download_streams_to_disk(self) -> None:
        payload = b"x" * 100_000 + b"tail"
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "f.bin")
            with mock.patch("urllib.request.urlopen",
                            return_value=_FakeResponse(payload)):
                sources.download_file("http://example.invalid/f.bin", dest)
            with open(dest, "rb") as f:
                self.assertEqual(f.read(), payload)

    def test_http_error_becomes_runtime_error(self) -> None:
        err = urllib.error.HTTPError(
            "http://example.invalid/missing", 404, "Not Found", {}, None)
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("urllib.request.urlopen", side_effect=err):
                with self.assertRaisesRegex(RuntimeError, "404"):
                    sources.download_file("http://example.invalid/missing",
                                          os.path.join(tmp, "x"))


class ParquetTest(unittest.TestCase):
    def test_missing_pyarrow_points_at_mteb(self) -> None:
        with mock.patch.dict(sys.modules, {"pyarrow": None,
                                           "pyarrow.parquet": None}):
            with self.assertRaisesRegex(RuntimeError, "--source mteb"):
                sources.read_parquet_rows("/nonexistent.parquet")


class BenchCliTest(unittest.TestCase):
    def test_list_and_pull_usage(self) -> None:
        self.assertEqual(cli_main(["bench", "list"]), 0)
        self.assertEqual(cli_main(["bench", "pull"]), 1)  # --dataset missing

    def test_pull_end_to_end_with_fake_fetcher(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "scifact")
            with mock.patch("app.bench.sources.download_file",
                            side_effect=_fake_fetcher):
                rc = cli_main(["bench", "pull", "--dataset", "scifact",
                               "--dest", dest])
            self.assertEqual(rc, 0)
            self.assertTrue(os.path.isfile(
                os.path.join(dest, "corpus.jsonl")))


if __name__ == "__main__":
    unittest.main()
