"""Focused coverage for the ONNX granite embedding backend (app/embeddings.py).

Stdlib unittest only (no new harness). Run from the project root:

    .venv/bin/python -m unittest tests.test_embeddings -v

The fake-session tests below run fully offline. The live-model test needs a
one-time ~200MB download and runs only with PROCURE_EMBEDDING_LIVE=1:

    PROCURE_EMBEDDING_LIVE=1 .venv/bin/python -m unittest tests.test_embeddings
"""
from __future__ import annotations

import os
import unittest

import numpy as np

from app.embeddings import (BACKEND, DIM, GraniteEmbedder,
                            parse_provider_names, resolve_providers)


class FakeEncoding:
    def __init__(self, ids: list[int], mask: list[int]) -> None:
        self.ids = ids
        self.attention_mask = mask


class FakeTokenizer:
    def __init__(self, pad_to: int = 0) -> None:
        self.pad_to = pad_to
        self.calls = 0

    def encode_batch(self, texts: list[str]) -> list[FakeEncoding]:
        self.calls += 1
        out = []
        for t in texts:
            ids = [1] + [ord(c) % 100 + 10 for c in t[:8]] + [2]
            out.append(FakeEncoding(ids, [1] * len(ids)))
        width = self.pad_to or max(len(e.ids) for e in out)
        for e in out:
            pad = width - len(e.ids)
            e.ids += [0] * pad
            e.attention_mask += [0] * pad
        return out


class FakeSession:
    def __init__(self, hidden: np.ndarray, want: set[str] | None = None) -> None:
        self.hidden = hidden
        self.want = want
        self.runs = 0
        self.feeds: list[dict] = []

    def run(self, _outputs, feed: dict):
        self.runs += 1
        self.feeds.append(feed)
        if self.want is not None:
            assert set(feed) == self.want, f"feed {sorted(feed)} != {sorted(self.want)}"
        width = len(next(iter(feed.values())))
        return [self.hidden[:width]]


def _wired(texts: list[str], hidden: np.ndarray,
           batch_size: int = 32,
           inputs: tuple[str, ...] = ("input_ids", "attention_mask"),
           want: set[str] | None = None) -> tuple[GraniteEmbedder, list]:
    e = GraniteEmbedder(batch_size=batch_size)
    e._session = FakeSession(hidden, want)  # type: ignore[assignment]
    e._tokenizer = FakeTokenizer()  # type: ignore[assignment]
    e._input_names = inputs
    return e, e.embed(texts)


class PoolingTest(unittest.TestCase):
    def test_cls_slice_l2_normalized(self) -> None:
        hidden = np.zeros((2, 4, 4), dtype=np.float32)
        hidden[0, 0] = [3.0, 0.0, 4.0, 0.0]  # norm 5
        hidden[1, 0] = [0.0, 0.0, 0.0, 2.0]  # norm 2
        hidden[:, 1:, :] = 999.0  # non-CLS tokens must be ignored
        _, vecs = _wired(["a", "b"], hidden)
        self.assertEqual(len(vecs), 2)
        for v in vecs:
            self.assertEqual(len(v), 4)
            self.assertAlmostEqual(float(np.linalg.norm(v)), 1.0)
        self.assertAlmostEqual(vecs[0][0], 0.6)
        self.assertAlmostEqual(vecs[0][2], 0.8)
        self.assertAlmostEqual(vecs[1][3], 1.0)

    def test_empty_input_needs_no_model(self) -> None:
        e = GraniteEmbedder()
        self.assertEqual(e.embed([]), [])
        self.assertFalse(e.loaded)

    def test_batches_split_on_batch_size(self) -> None:
        hidden = np.ones((5, 2, 4), dtype=np.float32)
        e, vecs = _wired(["t%d" % i for i in range(5)], hidden, batch_size=2)
        self.assertEqual(len(vecs), 5)
        self.assertEqual(e._session.runs, 3)

    def test_token_type_ids_fed_only_when_graph_wants_them(self) -> None:
        hidden = np.ones((1, 2, 4), dtype=np.float32)
        e, _ = _wired(
            ["x"], hidden,
            inputs=("input_ids", "attention_mask", "token_type_ids"),
            want={"input_ids", "attention_mask", "token_type_ids"})
        feed = e._session.feeds[0]
        self.assertTrue((feed["token_type_ids"] == 0).all())

    def test_max_length_clamped_to_model_context(self) -> None:
        self.assertEqual(GraniteEmbedder(max_length=100_000).max_length, 8192)
        self.assertEqual(GraniteEmbedder(max_length=512).max_length, 512)


class ProviderSelectionTest(unittest.TestCase):
    def test_prefers_gpu_over_cpu(self) -> None:
        self.assertEqual(
            resolve_providers(["CPUExecutionProvider",
                               "CUDAExecutionProvider"])[0],
            "CUDAExecutionProvider")

    def test_order_cuda_coreml_dml_cpu(self) -> None:
        avail = ["CPUExecutionProvider", "DmlExecutionProvider",
                 "CoreMLExecutionProvider", "CUDAExecutionProvider"]
        self.assertEqual(
            resolve_providers(avail),
            ["CUDAExecutionProvider", "CoreMLExecutionProvider",
             "DmlExecutionProvider", "CPUExecutionProvider"])

    def test_requested_come_first(self) -> None:
        avail = ["CPUExecutionProvider", "CUDAExecutionProvider",
                 "CoreMLExecutionProvider"]
        self.assertEqual(
            resolve_providers(avail, ("CoreMLExecutionProvider",)),
            ["CoreMLExecutionProvider", "CUDAExecutionProvider",
             "CPUExecutionProvider"])

    def test_tensorrt_only_when_requested(self) -> None:
        avail = ["CPUExecutionProvider", "TensorRTExecutionProvider",
                 "CUDAExecutionProvider"]
        default = resolve_providers(avail)
        self.assertNotIn("TensorRTExecutionProvider", default)
        self.assertEqual(
            resolve_providers(avail, ("TensorRTExecutionProvider",))[0],
            "TensorRTExecutionProvider")

    def test_cpu_always_last_resort(self) -> None:
        self.assertEqual(resolve_providers([]), ["CPUExecutionProvider"])
        self.assertEqual(
            resolve_providers(["WeirdExecutionProvider"]),
            ["WeirdExecutionProvider"])

    def test_aliases_and_unknowns(self) -> None:
        self.assertEqual(parse_provider_names("cuda, cpu"),
                         ("CUDAExecutionProvider", "CPUExecutionProvider"))
        self.assertEqual(parse_provider_names("dml"),
                         ("DmlExecutionProvider",))
        self.assertEqual(parse_provider_names("bogus"), ())
        self.assertEqual(parse_provider_names(None), ())

    def test_info_before_load_loads_nothing(self) -> None:
        e = GraniteEmbedder()
        self.assertFalse(e.loaded)
        self.assertIsNone(e.provider)
        info = e.info()
        self.assertEqual(info["dim"], DIM)
        self.assertEqual(info["backend"], BACKEND)
        self.assertEqual(info["providers_active"], [])


class LiveModelTest(unittest.TestCase):
    """End-to-end over the real ONNX graph; needs the model download."""

    @unittest.skipUnless(os.environ.get("PROCURE_EMBEDDING_LIVE") == "1",
                         "needs PROCURE_EMBEDDING_LIVE=1 (downloads ~200MB)")
    def test_real_model_semantics(self) -> None:
        e = GraniteEmbedder()
        docs = ["The north beacon was relamped in March.",
                "Harbor tide tables for April."]
        vecs = e.embed(docs)
        self.assertEqual(len(vecs[0]), 384)
        for v in vecs:
            self.assertAlmostEqual(float(np.linalg.norm(v)), 1.0, places=5)
        # Cosine = dot product on normalized vectors: the beacon query must
        # rank the beacon doc above the tide doc.
        q = np.asarray(e.embed(["beacon relamped"])[0])
        sims = [float(q @ np.asarray(v)) for v in vecs]
        self.assertGreater(sims[0], sims[1])
        self.assertTrue(e.loaded)
        import onnxruntime as ort

        available = ort.get_available_providers()
        self.assertIn(e.provider, available)
        gpu = [p for p in available if p != "CPUExecutionProvider"]
        if gpu:  # GPU present: auto-select must use it, never CPU
            self.assertIn(e.provider, gpu)


if __name__ == "__main__":
    unittest.main()
