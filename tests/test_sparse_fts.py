"""Regression coverage for the FTS5-backed sparse store (app/search.py).

Stdlib unittest only (no new harness). Run from the project root:

    .venv/bin/python -m unittest discover -s tests
"""
from __future__ import annotations

import math
import os
import sqlite3
import tempfile
import unittest

from app.search import SparseStore, content_terms
from app.config import Settings
from app.service import RAGService
from stub_embedder import StubEmbedder


def _settings(tmp: str) -> Settings:
    s = Settings()
    s.data_dir = tmp
    return s


class SparseFtsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = SparseStore(os.path.join(self.tmp.name, "sparse.sqlite"))
        self.store.upsert_chunks(
            ["c1", "c2", "c3"],
            ["d1", "d1", "d2"],
            ["Beacon beacon beacon relamped beacon maintenance log.",
             "Harbor tide tables for April. Ferries follow the morning tide.",
             "Beacon tide harbor relamped."],
        )

    def test_ranking_orders(self) -> None:
        # Orders verified against the previous hand-rolled BM25 (K1=1.5,
        # B=0.75) on this corpus: identical ranking, only the score scale
        # changed (FTS5 bm25 negated to larger-is-better positives).
        self.assertEqual([h.chunk_id for h in self.store.search("beacon relamped", 5)],
                         ["c1", "c3"])
        self.assertEqual([h.chunk_id for h in self.store.search("beacon tide", 5)],
                         ["c3", "c1", "c2"])
        self.assertEqual([h.chunk_id for h in self.store.search("tide", 5)],
                         ["c2", "c3"])
        hits = self.store.search("beacon relamped", 5)
        self.assertGreater(hits[0].score, 0.0)
        self.assertGreater(hits[0].score, hits[1].score)

    def test_top_n_and_empty_query(self) -> None:
        self.assertEqual(len(self.store.search("beacon tide", 1)), 1)
        self.assertEqual(self.store.search("", 5), [])
        self.assertEqual(self.store.search("*** (((", 5), [])
        self.assertEqual(self.store.search("zzz-no-match", 5), [])

    def test_fts_syntax_in_query_is_quoted_not_executed(self) -> None:
        # Reserved words / operators in the raw query must not raise and
        # must not change semantics beyond the extracted content terms.
        plain = [h.chunk_id for h in self.store.search("beacon tide", 5)]
        tricky = [h.chunk_id for h in self.store.search('beacon OR tide', 5)]
        self.assertEqual(tricky, plain)
        quoted = self.store.search('beacon "tide" * (', 5)
        self.assertEqual([h.chunk_id for h in quoted], plain)
        self.assertEqual(content_terms('a"b'), ["b"])

    def test_doc_scope_filter(self) -> None:
        hits = self.store.search("beacon", 5, doc_ids=["d2"])
        self.assertEqual([h.chunk_id for h in hits], ["c3"])
        self.assertEqual(hits[0].doc_id, "d2")

    def test_upsert_replaces_chunk(self) -> None:
        self.store.upsert_chunks(["c1"], ["d1"], ["sparrows gardening note"])
        self.assertEqual(self.store.count(), 3)
        self.assertEqual([h.chunk_id for h in self.store.search("beacon", 5)],
                         ["c3"])
        self.assertEqual([h.chunk_id for h in self.store.search("sparrows", 5)],
                         ["c1"])

    def test_delete_by_doc(self) -> None:
        self.assertEqual(self.store.delete_by_doc("d1"), 2)
        self.assertEqual(self.store.count(), 1)
        self.assertEqual(self.store.delete_by_doc("d1"), 0)
        self.assertEqual([h.chunk_id for h in self.store.search("beacon", 5)],
                         ["c3"])

    def test_idf_map_formula(self) -> None:
        idf = self.store.idf_map(["beacon", "zzz"])
        # N=3: beacon in 2 chunks, zzz in 0.
        self.assertAlmostEqual(idf["beacon"], math.log(1.0 + (3 - 2 + 0.5) / 2.5))
        self.assertAlmostEqual(idf["zzz"], math.log(1.0 + (3 - 0 + 0.5) / 0.5))
        self.assertEqual(self.store.idf_map([]), {})
        empty = SparseStore(os.path.join(self.tmp.name, "empty.sqlite"))
        self.assertEqual(empty.idf_map(["beacon"]), {})
        self.assertEqual(empty.search("beacon", 5), [])

    def test_legacy_schema_migrates(self) -> None:
        path = os.path.join(self.tmp.name, "legacy.sqlite")
        c = sqlite3.connect(path)
        c.execute(
            "CREATE TABLE chunks (chunk_id TEXT PRIMARY KEY, doc_id TEXT NOT NULL,"
            " text TEXT NOT NULL, length INTEGER NOT NULL DEFAULT 0)")
        c.execute(
            "CREATE TABLE postings (chunk_id TEXT NOT NULL, term TEXT NOT NULL,"
            " tf INTEGER NOT NULL, PRIMARY KEY (chunk_id, term))")
        c.execute("INSERT INTO chunks VALUES ('c1', 'd1', 'north beacon relamped', 3)")
        c.execute("INSERT INTO postings VALUES ('c1', 'beacon', 1)")
        c.execute("CREATE INDEX idx_post_term ON postings(term)")
        c.execute("CREATE INDEX idx_chunks_doc ON chunks(doc_id)")
        c.commit()
        c.close()
        migrated = SparseStore(path)
        self.assertEqual(migrated.count(), 1)
        hits = migrated.search("beacon", 5)
        self.assertEqual([(h.chunk_id, h.text) for h in hits],
                         [("c1", "north beacon relamped")])
        tables = {r[0] for r in sqlite3.connect(path).execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        self.assertNotIn("chunks", tables)
        self.assertNotIn("postings", tables)
        # Reopen is idempotent (no duplication, still searchable).
        self.assertEqual(SparseStore(path).count(), 1)

    def test_service_search_uses_sparse_channel(self) -> None:
        svc = RAGService(_settings(os.path.join(self.tmp.name, "data")),
                         embedder=StubEmbedder())
        svc.ingest_text("a.md", "The north beacon was relamped. " * 20)
        hits = svc.search("beacon relamped")
        self.assertTrue(hits)
        self.assertGreaterEqual(hits[0]["sparse_score"], 0.0)


if __name__ == "__main__":
    unittest.main()
