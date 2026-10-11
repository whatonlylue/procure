"""OnnxCrossEncoder head math and CrossEncoderScorer lifecycle (no network)."""

from __future__ import annotations

import json
import math
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from safetensors.numpy import save_file

from app import rerank_onnx
from app.rerank_onnx import OnnxCrossEncoder
from app.search import CrossEncoderScorer


class _Enc:
    def __init__(self, n):
        self.ids = list(range(1, n + 1)) + [0] * (4 - n)
        self.attention_mask = [1] * n + [0] * (4 - n)
        self.type_ids = [0] * 4


class _Tok:
    def encode_batch(self, pairs):
        return [_Enc(2 + (i % 2)) for i, _ in enumerate(pairs)]


class _Input:
    def __init__(self, name):
        self.name = name


class _Sess:
    def __init__(self, hidden, names=("input_ids", "attention_mask")):
        self.hidden = hidden
        self.names = names
        self.feeds = None

    def get_inputs(self):
        return [_Input(n) for n in self.names]

    def run(self, _out, feeds):
        self.feeds = feeds
        n = feeds["input_ids"].shape[0]
        return [self.hidden[:n]]


def _make_dir(root: Path, pooling="cls") -> None:
    (root / "1_Pooling").mkdir()
    (root / "2_Dense").mkdir()
    (root / "3_LayerNorm").mkdir()
    (root / "4_Dense").mkdir()
    (root / "modules.json").write_text(json.dumps([
        {"idx": 0, "path": "", "type": "x.Transformer"},
        {"idx": 1, "path": "1_Pooling", "type": "x.Pooling"},
        {"idx": 2, "path": "2_Dense", "type": "x.Dense"},
        {"idx": 3, "path": "3_LayerNorm", "type": "x.LayerNorm"},
        {"idx": 4, "path": "4_Dense", "type": "x.Dense"},
    ]))
    (root / "1_Pooling/config.json").write_text(json.dumps({"pooling_mode": pooling}))
    (root / "2_Dense/config.json").write_text(json.dumps(
        {"activation_function": "torch.nn.modules.activation.GELU"}))
    (root / "3_LayerNorm/config.json").write_text("{}")
    (root / "4_Dense/config.json").write_text(json.dumps(
        {"activation_function": "torch.nn.modules.linear.Identity"}))


def _weights():
    rng = np.random.default_rng(0)
    return {
        "w2": rng.normal(size=(3, 3)).astype(np.float32),
        "lw": rng.normal(size=3).astype(np.float32),
        "lb": rng.normal(size=3).astype(np.float32),
        "w4": rng.normal(size=(1, 3)).astype(np.float32),
        "b4": np.array([0.25], dtype=np.float32),
    }


def _hand(h, w):
    h = h.astype(np.float64) @ w["w2"].T.astype(np.float64)
    h = np.array([[v * 0.5 * (1 + math.erf(v / math.sqrt(2))) for v in r] for r in h])
    mu, var = h.mean(-1, keepdims=True), h.var(-1, keepdims=True)
    h = (h - mu) / np.sqrt(var + 1e-5) * w["lw"] + w["lb"]
    return (h @ w["w4"].T + w["b4"])[:, 0]


def _build(root: Path, hidden, pooling="cls") -> OnnxCrossEncoder:
    _make_dir(root, pooling)
    w = _weights()
    save_file({"linear.weight": w["w2"]}, str(root / "2_Dense/model.safetensors"))
    save_file({"norm.weight": w["lw"], "norm.bias": w["lb"]},
              str(root / "3_LayerNorm/model.safetensors"))
    save_file({"linear.weight": w["w4"], "linear.bias": w["b4"]},
              str(root / "4_Dense/model.safetensors"))
    files = {p.relative_to(root).as_posix(): p for p in root.rglob("*") if p.is_file()}
    files["m.onnx"] = root / "m.onnx"
    files["tokenizer.json"] = root / "tokenizer.json"
    enc = OnnxCrossEncoder.__new__(OnnxCrossEncoder)
    enc.repo_id, enc.max_length, enc._steps = "fake", 512, []
    sess = _Sess(hidden)
    with mock.patch("onnxruntime.InferenceSession", return_value=sess), \
            mock.patch("tokenizers.Tokenizer.from_file") as ff:
        ff.return_value = mock.MagicMock()
        enc._build(files, "m.onnx")
    enc._tok, enc._sess, enc._input_names = _Tok(), sess, sess.get_inputs() and \
        [i.name for i in sess.get_inputs()]
    return enc


class HeadTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.hidden = np.random.default_rng(1).normal(size=(2, 4, 3)).astype(np.float32)

    def test_cls_head_matches_hand_computation(self):
        enc = _build(self.root, self.hidden)
        got = enc.score("q", ["a", "b"])
        want = _hand(self.hidden[:, 0], _weights())
        np.testing.assert_allclose(got, want, atol=1e-6)
        self.assertEqual(set(enc._sess.feeds), {"input_ids", "attention_mask"})
        self.assertEqual(enc._sess.feeds["input_ids"].dtype, np.int64)

    def test_mean_pooling_respects_mask(self):
        enc = _build(self.root, self.hidden, pooling="mean")
        got = enc.score("q", ["a", "b"])
        # row 0 has 2 real tokens, row 1 has 3
        pooled = np.stack([self.hidden[0, :2].mean(0), self.hidden[1, :3].mean(0)])
        np.testing.assert_allclose(got, _hand(pooled, _weights()), atol=1e-6)

    def test_scalar_output_passthrough(self):
        enc = _build(self.root, np.zeros((2, 4, 3), np.float32))
        enc._steps = []
        enc._sess.hidden = np.array([[1.5], [-2.0]], dtype=np.float32)
        self.assertEqual(enc.score("q", ["a", "b"]), [1.5, -2.0])

    def test_erf_gelu_values(self):
        x = np.array([-2.0, 0.0, 1.0])
        want = [v * 0.5 * (1 + math.erf(v / math.sqrt(2))) for v in x]
        np.testing.assert_allclose(rerank_onnx._gelu(x), want, atol=1e-12)


class ScorerTest(unittest.TestCase):
    def test_background_load_then_scores(self):
        gate = threading.Event()

        class Fake:
            def __init__(self, *a, **k):
                gate.wait(5)

            def score(self, q, texts):
                return [float(len(t)) for t in texts]

        with mock.patch("app.rerank_onnx.OnnxCrossEncoder", Fake):
            s = CrossEncoderScorer("fake/model", True)
            self.assertIsNone(s.score("q", ["abc"]))  # starts loading
            self.assertFalse(s.loaded)
            gate.set()
            self.assertTrue(s.warmup())
            self.assertEqual(s.score("q", ["abc", "de"]), [3.0, 2.0])
            self.assertTrue(s.available)

    def test_warmup_sync(self):
        class Fake:
            def __init__(self, *a, **k):
                pass

            def score(self, q, texts):
                return [1.0] * len(texts)

        with mock.patch("app.rerank_onnx.OnnxCrossEncoder", Fake):
            s = CrossEncoderScorer("fake/model", True)
            self.assertTrue(s.warmup())
            self.assertEqual(s.score("q", ["x"]), [1.0])

    def test_load_failure_returns_none_and_logs_once(self):
        calls = []

        def boom(*a, **k):
            calls.append(1)
            raise OSError("offline")

        with mock.patch("app.rerank_onnx.OnnxCrossEncoder", boom), \
                self.assertLogs("app.search", "WARNING") as cm:
            s = CrossEncoderScorer("fake/model", False)
            self.assertFalse(s.warmup())
            self.assertIsNone(s.score("q", ["x"]))
            self.assertIsNone(s.score("q", ["x"]))
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(cm.records), 1)

    def test_empty_model_name_disabled(self):
        s = CrossEncoderScorer("", True)
        self.assertFalse(s.warmup())
        self.assertIsNone(s.score("q", ["x"]))


if __name__ == "__main__":
    unittest.main()
