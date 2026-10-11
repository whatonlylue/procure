"""Rowid map, tokenizer migration, convex fusion and rerank blend tests."""
from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest

from app.search import (
    DENSE_WEIGHT,
    SparseStore,
    combine_answerability,
    convex_fuse,
)
from app.store import close_keepalives

_OLD_CREATE = (
    "CREATE VIRTUAL TABLE chunks_fts USING fts5("
    "chunk_id UNINDEXED, doc_id UNINDEXED, text, "
    "tokenize='unicode61 remove_diacritics 2')"
)


def _fts_rows(path: str) -> list[tuple]:
    c = sqlite3.connect(path)
    try:
        return c.execute(
            "SELECT rowid, chunk_id, doc_id, text FROM chunks_fts ORDER BY rowid"
        ).fetchall()
    finally:
        c.close()


class _TmpCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(close_keepalives)
        self.path = os.path.join(self.tmp.name, "sparse.sqlite")


class RowmapTest(_TmpCase):
    def test_delete_by_chunk_and_doc_remove_exact_rows(self) -> None:
        store = SparseStore(self.path)
        store.upsert_chunks(["a", "b", "c"], ["d1", "d1", "d2"],
                            ["alpha one", "beta two", "gamma three"])
        store.upsert_chunks(["a"], ["d1"], ["alpha replaced"])
        self.assertEqual([r[1] for r in _fts_rows(self.path)], ["b", "c", "a"])
        self.assertEqual(store.delete_by_doc("d1"), 2)
        self.assertEqual([r[1:3] for r in _fts_rows(self.path)], [("c", "d2")])
        self.assertEqual(store.delete_by_doc("d1"), 0)
        c = sqlite3.connect(self.path)
        self.addCleanup(c.close)
        self.assertEqual(c.execute(
            "SELECT chunk_id FROM fts_rowmap").fetchall(), [("c",)])

    def test_upsert_moves_chunk_between_docs(self) -> None:
        store = SparseStore(self.path)
        store.upsert_chunks(["a"], ["d1"], ["alpha"])
        store.upsert_chunks(["a"], ["d2"], ["alpha"])
        self.assertEqual(store.delete_by_doc("d1"), 0)
        self.assertEqual(store.delete_by_doc("d2"), 1)
        self.assertEqual(store.count(), 0)

    def test_backfill_from_fts_without_map(self) -> None:
        c = sqlite3.connect(self.path)
        c.execute(_OLD_CREATE)
        c.executemany(
            "INSERT INTO chunks_fts(chunk_id, doc_id, text) VALUES (?,?,?)",
            [("a", "d1", "alpha"), ("b", "d1", "beta"), ("a", "d1", "alpha dup"),
             ("c", "d2", "gamma")])
        c.commit()
        c.close()
        store = SparseStore(self.path)
        self.assertEqual(store.count(), 3)  # duplicate "a" collapsed
        self.assertEqual(store.delete_by_doc("d1"), 2)
        self.assertEqual([r[1] for r in _fts_rows(self.path)], ["c"])
        store.upsert_chunks(["c"], ["d2"], ["gamma again"])
        self.assertEqual(store.count(), 1)

    def test_legacy_tables_migrate_with_map(self) -> None:
        c = sqlite3.connect(self.path)
        c.execute("CREATE TABLE chunks(chunk_id TEXT, doc_id TEXT, text TEXT)")
        c.execute("CREATE TABLE postings(term TEXT, chunk_id TEXT)")
        c.execute("INSERT INTO chunks VALUES ('a','d1','alpha'),('b','d2','beta')")
        c.commit()
        c.close()
        store = SparseStore(self.path)
        self.assertEqual(store.count(), 2)
        self.assertEqual(store.delete_by_doc("d1"), 1)
        self.assertEqual([r[1] for r in _fts_rows(self.path)], ["b"])


class TokenizerMigrationTest(_TmpCase):
    def test_old_tokenizer_rebuilt_to_porter(self) -> None:
        c = sqlite3.connect(self.path)
        c.execute(_OLD_CREATE)
        c.executemany(
            "INSERT INTO chunks_fts(chunk_id, doc_id, text) VALUES (?,?,?)",
            [("a", "d1", "She runs daily"), ("b", "d2", "Unrelated text")])
        c.commit()
        before = c.execute(
            "SELECT rowid, chunk_id, doc_id, text FROM chunks_fts ORDER BY rowid"
        ).fetchall()
        c.close()
        store = SparseStore(self.path)
        self.assertEqual(_fts_rows(self.path), before)
        c = sqlite3.connect(self.path)
        self.addCleanup(c.close)
        sql = c.execute(
            "SELECT sql FROM sqlite_master WHERE name='chunks_fts'").fetchone()[0]
        self.assertIn("porter", sql)
        self.assertEqual([h.chunk_id for h in store.search("running", 5)], ["a"])
        # Idempotent: reopening leaves data and tokenizer intact.
        store2 = SparseStore(self.path)
        self.assertEqual(_fts_rows(self.path), before)
        self.assertEqual(store2.delete_by_doc("d1"), 1)

    def test_new_db_uses_porter(self) -> None:
        store = SparseStore(self.path)
        store.upsert_chunks(["a"], ["d"], ["she runs daily"])
        self.assertEqual([h.chunk_id for h in store.search("running", 5)], ["a"])


class FusionTest(unittest.TestCase):
    def test_convex_fuse_math_and_missing(self) -> None:
        out = convex_fuse({"a": 10.0, "b": 0.0}, {"b": 4.0, "c": 2.0}, 0.25)
        self.assertAlmostEqual(out["a"], 0.25)
        self.assertAlmostEqual(out["b"], 0.75)  # sparse 1.0, dense 0.0
        self.assertAlmostEqual(out["c"], 0.0)   # sparse min -> 0, no dense
        self.assertEqual(set(out), {"a", "b", "c"})

    def test_convex_fuse_defaults_and_degenerate(self) -> None:
        out = convex_fuse({"a": 0.5}, {})
        self.assertAlmostEqual(out["a"], DENSE_WEIGHT)
        self.assertEqual(convex_fuse({}, {}), {})
        self.assertAlmostEqual(convex_fuse({}, {"x": 3.0})["x"], 1 - DENSE_WEIGHT)


class AnswerabilityTest(unittest.TestCase):
    def test_minmax_over_logits(self) -> None:
        out = combine_answerability([-2.0, 3.0, 8.0], [0.9, 0.9, 0.9])
        self.assertEqual(out, [0.0, 0.5, 1.0])

    def test_all_equal_logits(self) -> None:
        self.assertEqual(combine_answerability([1.0, 1.0], [0.0, 0.0]), [1.0, 1.0])

    def test_coverage_fallback(self) -> None:
        cov = [0.2, 0.7]
        out = combine_answerability(None, cov)
        self.assertEqual(out, cov)
        self.assertIsNot(out, cov)


if __name__ == "__main__":
    unittest.main()
