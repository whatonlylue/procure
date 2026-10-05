"""Focused coverage for the model2vec embedding backend (app/embeddings.py).

Stdlib unittest only (no new harness). Run from the project root:

    .venv/bin/python -m unittest tests.test_embeddings -v

The fake-model tests below run fully offline. The live-model test needs a
one-time ~120MB download and runs only with PROCURE_EMBEDDING_LIVE=1:

    PROCURE_EMBEDDING_LIVE=1 .venv/bin/python -m unittest tests.test_embeddings
"""
from __future__ import annotations

import os
import unittest

import numpy as np

from app.embeddings import (BACKEND, DEVICE, DIM, MODEL_ID,
                            Model2VecEmbedder)


class FakeStaticModel:
    """Stands in for model2vec.StaticModel: deterministic per text."""

    def __init__(self, dim: int = DIM, scale: float = 3.0) -> None:
        self.dim = dim
        self.scale = scale
        self.calls: list[dict] = []

    def encode(self, sentences, show_progress_bar: bool = False,
               max_length=None, **kwargs):
        self.calls.append({"n": len(sentences), "max_length": max_length})
        vecs = np.zeros((len(sentences), self.dim), dtype=np.float32)
        for i, s in enumerate(sentences):
            seed = abs(hash(s)) % (2 ** 32)
            vecs[i] = (np.random.RandomState(seed).randn(self.dim)
                       .astype(np.float32) * self.scale)
        return vecs


def _wired(texts: list[str], batch_size: int = 32,
           max_length: int = 512, **kw) -> tuple[Model2VecEmbedder, list]:
    e = Model2VecEmbedder(batch_size=batch_size, max_length=max_length)
    e._model = FakeStaticModel(**kw)  # type: ignore[assignment]
    return e, e.embed(texts)


class BackendConstantsTest(unittest.TestCase):
    def test_potion_retrieval_defaults(self) -> None:
        self.assertEqual(MODEL_ID, "minishlab/potion-retrieval-32M")
        self.assertEqual(DIM, 512)
        self.assertEqual(BACKEND, "potion-retrieval-32m")
        # model2vec inference is CPU-only by design: no GPU to select.
        self.assertEqual(DEVICE, "cpu")
        self.assertEqual(Model2VecEmbedder.device, "cpu")


class EmbedBehaviorTest(unittest.TestCase):
    def test_output_l2_normalized_even_when_model_is_not(self) -> None:
        _, vecs = _wired(["a", "b"])
        self.assertEqual(len(vecs), 2)
        for v in vecs:
            self.assertEqual(len(v), DIM)
            self.assertAlmostEqual(float(np.linalg.norm(v)), 1.0)

    def test_empty_input_needs_no_model(self) -> None:
        e = Model2VecEmbedder()
        self.assertEqual(e.embed([]), [])
        self.assertFalse(e.loaded)

    def test_batches_split_on_batch_size(self) -> None:
        e, vecs = _wired(["t%d" % i for i in range(5)], batch_size=2)
        self.assertEqual(len(vecs), 5)
        self.assertEqual([c["n"] for c in e._model.calls], [2, 2, 1])

    def test_batching_does_not_change_vectors(self) -> None:
        texts = ["t%d" % i for i in range(5)]
        _, whole = _wired(texts, batch_size=32)
        _, split = _wired(texts, batch_size=2)
        self.assertEqual(whole, split)

    def test_non_strings_become_empty(self) -> None:
        e, vecs = _wired(["x", None, 42])  # type: ignore[list-item]
        self.assertEqual(len(vecs), 3)
        self.assertEqual([c["n"] for c in e._model.calls], [3])

    def test_max_length_reaches_model2vec(self) -> None:
        e, _ = _wired(["x"], max_length=128)
        self.assertEqual(e._model.calls[0]["max_length"], 128)
        self.assertEqual(Model2VecEmbedder(max_length=0).max_length, 1)

    def test_width_mismatch_rejected(self) -> None:
        import sys
        import types

        fake = types.ModuleType("model2vec")

        class WrongWidth:
            @classmethod
            def from_pretrained(cls, target):
                return FakeStaticModel(dim=256)

        fake.StaticModel = WrongWidth  # type: ignore[attr-defined]
        old = sys.modules.get("model2vec")
        sys.modules["model2vec"] = fake
        try:
            with self.assertRaises(RuntimeError):
                Model2VecEmbedder().embed(["x"])
        finally:
            if old is not None:
                sys.modules["model2vec"] = old
            else:
                del sys.modules["model2vec"]

    def test_missing_model_dir_fails_before_network(self) -> None:
        e = Model2VecEmbedder(model_dir="/nonexistent-procure-model-xyz")
        with self.assertRaises(RuntimeError):
            e.embed(["x"])
        self.assertFalse(e.loaded)

    def test_info_before_load_loads_nothing(self) -> None:
        e = Model2VecEmbedder()
        self.assertFalse(e.loaded)
        info = e.info()
        self.assertEqual(info["dim"], DIM)
        self.assertEqual(info["backend"], BACKEND)
        self.assertEqual(info["device"], "cpu")
        self.assertFalse(info["loaded"])


class SidecarSpecTest(unittest.TestCase):
    """The frozen sidecar must bundle every load-time dependency of the
    embedding backend.

    Regression: ``safetensors`` was once excluded to keep torch out of
    the binary, but model2vec imports it lazily at model-load time, so
    the frozen app failed with ``No module named 'safetensors'`` while
    the source tree worked fine.
    """

    SPEC = os.path.join(os.path.dirname(__file__), "..",
                        "procure-sidecar.spec")

    def _excludes(self) -> set[str]:
        import ast

        with open(self.SPEC, encoding="utf-8") as f:
            tree = ast.parse(f.read())
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and getattr(node.func, "id", "") == "Analysis"):
                for kw in node.keywords:
                    if kw.arg == "excludes" and isinstance(
                            kw.value, ast.List):
                        return {e.value for e in kw.value.elts
                                if isinstance(e, ast.Constant)}
        self.fail("no excludes list found in procure-sidecar.spec")

    def test_embedding_deps_not_excluded(self) -> None:
        self.assertTrue(os.path.isfile(self.SPEC))
        excluded = self._excludes()
        # app.embeddings hard deps plus model2vec's own load-time deps.
        for mod in ("model2vec", "numpy", "safetensors", "tokenizers",
                    "huggingface_hub", "joblib", "tqdm"):
            self.assertNotIn(mod, excluded,
                             f"{mod} must ship in the sidecar")


class LiveModelTest(unittest.TestCase):
    """End-to-end over the real model2vec model; needs the download."""

    @unittest.skipUnless(os.environ.get("PROCURE_EMBEDDING_LIVE") == "1",
                         "needs PROCURE_EMBEDDING_LIVE=1 (downloads ~120MB)")
    def test_real_model_semantics(self) -> None:
        e = Model2VecEmbedder()
        docs = ["The north beacon was relamped in March.",
                "Harbor tide tables for April."]
        vecs = e.embed(docs)
        self.assertEqual(len(vecs[0]), 512)
        for v in vecs:
            self.assertAlmostEqual(float(np.linalg.norm(v)), 1.0, places=5)
        # Cosine = dot product on normalized vectors: the beacon query must
        # rank the beacon doc above the tide doc.
        q = np.asarray(e.embed(["beacon relamped"])[0])
        sims = [float(q @ np.asarray(v)) for v in vecs]
        self.assertGreater(sims[0], sims[1])
        self.assertTrue(e.loaded)
        self.assertEqual(e.info()["device"], "cpu")


if __name__ == "__main__":
    unittest.main()
