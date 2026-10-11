import os
import tempfile
import unittest

import numpy as np

from app.store import close_keepalives
from app.vectordb.sqlite_vec import SqliteVectorStore


def _key(hits):
    return [(h.chunk_id, h.doc_id, h.text) for h in hits]


class VectorCacheTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.addCleanup(close_keepalives)
        self.path = os.path.join(tmp.name, "v.db")
        self.stores: list[SqliteVectorStore] = []

    def _store(self) -> SqliteVectorStore:
        s = SqliteVectorStore(self.path)
        self.addCleanup(s.close)
        self.stores.append(s)
        return s

    def _fill(self, s, n=60, dim=16, seed=0):
        rng = np.random.default_rng(seed)
        vecs = rng.normal(size=(n, dim)).astype(np.float32)
        vecs[3] = 0.0
        vecs[10] = vecs[11]  # exact tie
        ids = [f"d{i % 5}:{i}" for i in range(n)]
        dids = [f"d{i % 5}" for i in range(n)]
        s.upsert(ids, dids, vecs.tolist(), [f"t{i}" for i in range(n)])
        return rng

    def _assert_same(self, s, q, k, doc_ids=None):
        new = s.search(q, k, doc_ids)
        old = s._search_uncached(q, k, doc_ids)
        self.assertEqual(_key(new), _key(old))
        np.testing.assert_allclose([h.score for h in new],
                                   [h.score for h in old], atol=1e-5)

    def test_matches_uncached(self):
        s = self._store()
        rng = self._fill(s)
        for _ in range(5):
            q = rng.normal(size=16).tolist()
            self._assert_same(s, q, 10)
            self._assert_same(s, q, 7, ["d2", "d0", "d2", "nope"])
            self._assert_same(s, q, 5, ["nope"])
        self._assert_same(s, [0.0] * 16, 5)

    def test_cross_instance_write_visible(self):
        a = self._store()
        self._fill(a)
        q = [1.0] + [0.0] * 15
        a.search(q, 3)  # warm cache
        b = self._store()
        b.upsert(["x:0"], ["x"], [[1.0] + [0.0] * 15], ["fresh"])
        hits = a.search(q, 1)
        self.assertEqual(hits[0].chunk_id, "x:0")
        b.delete_by_doc("x")
        self.assertNotIn("x:0", [h.chunk_id for h in a.search(q, 100)])

    def test_delete_by_doc(self):
        s = self._store()
        rng = self._fill(s)
        q = rng.normal(size=16).tolist()
        s.search(q, 5)
        s.delete_by_doc("d1")
        hits = s.search(q, 100)
        self.assertTrue(hits)
        self.assertNotIn("d1", {h.doc_id for h in hits})
        self.assertEqual(s.search(q, 5, ["d1"]), [])

    def test_mixed_widths_fall_back(self):
        s = self._store()
        s.upsert(["a:0", "a:1"], ["a", "a"], [[1.0, 0.0], [0.0, 1.0]], ["x", "y"])
        s.upsert(["b:0"], ["b"], [[1.0, 0.0, 0.0]], ["z"])
        hits = s.search([1.0, 0.0], 5)
        self.assertEqual(s.last_skipped, 1)
        self.assertEqual(_key(hits), _key(s._search_uncached([1.0, 0.0], 5)))
        hits = s.search([1.0, 0.0, 0.0], 5)
        self.assertEqual(s.last_skipped, 2)
        self.assertEqual([h.chunk_id for h in hits], ["b:0"])

    def test_query_dim_mismatch_counts_skipped(self):
        s = self._store()
        self._fill(s, n=12)
        self.assertEqual(s.search([1.0, 0.0], 3), [])
        self.assertEqual(s.last_skipped, 12)

    def test_close_twice(self):
        s = self._store()
        s.close()
        s.close()


if __name__ == "__main__":
    unittest.main()
