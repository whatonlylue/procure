"""Granite embedder plumbing, backend switching, re-embed job, CE retry.

Fully offline: the ONNX session is faked and no real model is ever built.
"""
from __future__ import annotations

import tempfile
import unittest
from unittest import mock

import numpy as np

from app import search as search_mod
from app.config import Settings
from app.embeddings import (GRANITE_BACKEND, GRANITE_DIM, Model2VecEmbedder,
                            OnnxEmbedder, build_embedder)
from app.jobs import JobManager
from app.search import CrossEncoderScorer
from app.service import RAGService
from stub_embedder import StubEmbedder


class _Enc:
    def __init__(self, ids):
        self.ids = ids


class _FakeTok:
    def encode_batch(self, texts):
        return [_Enc(list(range(1, 1 + max(1, len(t.split()))))) for t in texts]


class _FakeSess:
    def __init__(self):
        self.shapes = []

    def run(self, _out, feeds):
        ids = feeds["input_ids"]
        self.shapes.append(ids.shape)
        h = np.zeros((ids.shape[0], ids.shape[1], GRANITE_DIM), np.float32)
        # CLS vector encodes the row's first real length, so order is checkable
        h[:, 0, 0] = feeds["attention_mask"].sum(axis=1)
        h[:, 0, 1] = 1.0
        return [h]


def _fake_granite(batch=2) -> OnnxEmbedder:
    e = OnnxEmbedder(batch_size=batch)
    e._tok, e._sess, e._input_names = _FakeTok(), _FakeSess(), [
        "input_ids", "attention_mask"]
    return e


class GraniteEmbedderTest(unittest.TestCase):
    def test_order_normalization_and_length_batching(self) -> None:
        e = _fake_granite(batch=2)
        texts = ["a b c d", "a", "a b", "a b c", ""]
        out = e.embed(texts)
        self.assertEqual(len(out), 5)
        for v in out:
            self.assertEqual(len(v), GRANITE_DIM)
            self.assertAlmostEqual(float(np.linalg.norm(v)), 1.0, places=5)
        # order restored: component 0/1 ratio == token count
        lens = [4, 1, 2, 3, 1]
        for v, n in zip(out, lens):
            self.assertAlmostEqual(v[0] / v[1], n, places=4)
        # sorted by length -> padding within a batch is minimal
        self.assertLessEqual(max(s[1] for s in e._sess.shapes), 4)
        self.assertEqual(e._sess.shapes[0][1], 1)  # shortest batch first

    def test_metadata_and_cheap_info(self) -> None:
        e = OnnxEmbedder()
        self.assertEqual(e.backend, GRANITE_BACKEND)
        self.assertEqual(e.fusion_dense_weight, 0.7)
        self.assertFalse(e.info()["loaded"])
        self.assertEqual(Model2VecEmbedder().fusion_dense_weight, 0.3)

    def test_build_embedder_selection(self) -> None:
        s = Settings()
        s.embedding_backend = "granite"
        self.assertIsInstance(build_embedder(s), OnnxEmbedder)
        s.embedding_backend = "potion"
        self.assertIsInstance(build_embedder(s), Model2VecEmbedder)


class SwitchAndReembedTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        s = Settings()
        s.data_dir = self.tmp.name
        self.svc = RAGService(s, embedder=StubEmbedder("old-backend"))
        for i in range(5):
            self.svc.ingest_text(f"n{i}.txt", f"Lighthouse note number {i}. " * 80)

    def _switch_to_fake_granite(self) -> OnnxEmbedder:
        with mock.patch("app.service.build_embedder") as b:
            fake = _fake_granite()
            b.return_value = fake
            r = self.svc.update_settings(embedding_backend="granite")
        self.assertTrue(r["embedding_switched"])
        return fake

    def test_settings_persist_and_validate(self) -> None:
        with self.assertRaises(ValueError):
            self.svc.update_settings(embedding_backend="gpt")
        self._switch_to_fake_granite()
        s2 = Settings()
        s2.data_dir = self.tmp.name
        s2._load_overrides()
        self.assertEqual(s2.embedding_backend, "granite")
        again = self.svc.update_settings(embedding_backend="granite")
        self.assertFalse(again["embedding_switched"])

    def test_reembed_switches_tag_only_at_end_and_midway_is_safe(self) -> None:
        fake = self._switch_to_fake_granite()
        n = self.svc.vectors.count()
        self.assertEqual(self.svc.vectors.stored_backend(), "old-backend")
        seen = []
        real = self.svc.vectors.replace_embeddings

        def spy(ids, embs):
            real(ids, embs)
            # half-way: old tag still recorded, mixed widths in the store
            seen.append((self.svc.vectors.stored_backend(),
                         self.svc.vectors.stored_dim()))
            hits = self.svc.search("lighthouse note", 3)  # must not raise
            self.assertTrue(hits)
        with mock.patch.object(self.svc.vectors, "replace_embeddings", spy):
            res = self.svc.reembed_all()
        self.assertEqual(res["status"], "done")
        self.assertEqual(res["done"], n)
        self.assertTrue(all(b == "old-backend" for b, _ in seen))
        self.assertEqual(self.svc.vectors.stored_backend(), GRANITE_BACKEND)
        self.assertEqual(self.svc.vectors.stored_dim(), GRANITE_DIM)
        ds = self.svc.dimension_status()
        self.assertTrue(ds["dense_ok"], ds)
        hits = self.svc.search("lighthouse note", 3)
        self.assertTrue(all(h["dense_score"] != 0.0 for h in hits))
        self.assertIs(self.svc.embedder, fake)

    def test_mixed_width_store_search_skips_stale(self) -> None:
        self._switch_to_fake_granite()
        rows = self.svc.vectors.all_chunks()[:2]
        embs = self.svc.embedder.embed([r[2] for r in rows])
        self.svc.vectors.replace_embeddings([r[0] for r in rows], embs)
        self.svc.vectors.set_corpus_meta(GRANITE_DIM, GRANITE_BACKEND)
        hits = self.svc.vectors.search(embs[0], 10)
        self.assertEqual(len(hits), 2)
        self.assertEqual(self.svc.vectors.last_skipped,
                         self.svc.vectors.count() - 2)

    def test_cancel_keeps_old_tag(self) -> None:
        self._switch_to_fake_granite()
        res = self.svc.reembed_all(cancelled=lambda: True)
        self.assertEqual(res["status"], "cancelled")
        self.assertEqual(self.svc.vectors.stored_backend(), "old-backend")

    def test_replace_embeddings_never_inserts(self) -> None:
        n = self.svc.vectors.count()
        self.svc.vectors.replace_embeddings(["ghost:0"], [[1.0, 0.0]])
        self.assertEqual(self.svc.vectors.count(), n)

    def test_fusion_weight_follows_embedder(self) -> None:
        self._switch_to_fake_granite()
        self.svc.reembed_all()
        with mock.patch("app.service.convex_fuse",
                        wraps=search_mod.convex_fuse) as cf:
            self.svc.search("lighthouse note", 3)
        self.assertEqual(cf.call_args[0][2], 0.7)

    def test_job_status_via_manager(self) -> None:
        self._switch_to_fake_granite()
        jm = JobManager()
        self.addCleanup(jm.shutdown)
        job = jm.submit("reembed", "x", lambda h: self.svc.reembed_all(
            progress=h.update, cancelled=h.is_cancelled))
        done = jm.wait(job["job_id"], 30)
        self.assertEqual(done["status"], "done")
        self.assertEqual(done["done"], done["total"])


class CrossEncoderRetryTest(unittest.TestCase):
    def test_failed_load_retries_after_cooldown(self) -> None:
        sc = CrossEncoderScorer("some/model", True)
        calls = []

        class Boom:
            def __init__(self, *a, **k):
                calls.append(1)
                raise OSError("network down")
        with mock.patch("app.rerank_onnx.OnnxCrossEncoder", Boom):
            self.assertFalse(sc.warmup())
            self.assertFalse(sc.warmup())  # within cooldown: no retry
            self.assertEqual(len(calls), 1)
            sc._failed_at -= search_mod.CROSS_ENCODER_RETRY_SECONDS + 1
            self.assertFalse(sc.warmup())
            self.assertEqual(len(calls), 2)

        class Ok:
            def __init__(self, *a, **k):
                pass

            def score(self, q, texts):
                return [1.0] * len(texts)
        sc._failed_at -= search_mod.CROSS_ENCODER_RETRY_SECONDS + 1
        with mock.patch("app.rerank_onnx.OnnxCrossEncoder", Ok):
            self.assertIsNone(sc.score("q", ["t"]))  # kicks background load
            sc._thread.join() if sc._thread else None
            self.assertEqual(sc.score("q", ["t"]), [1.0])


if __name__ == "__main__":
    unittest.main()
