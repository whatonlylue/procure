"""Torch-free cross-encoder: ONNX backbone + numpy sentence-transformers head.

Loads a sentence-transformers CrossEncoder repo that ships an ONNX export of
the transformer backbone. The module stack after the backbone (Pooling, Dense,
LayerNorm, Dense ...) is read from ``modules.json`` and evaluated in numpy, so
only onnxruntime, tokenizers, numpy, safetensors and huggingface_hub are needed.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
from collections.abc import Iterator
from pathlib import Path

import numpy as np


@contextlib.contextmanager
def _hf_offline(allow_download: bool) -> Iterator[None]:
    """Force HuggingFace Hub offline unless downloads are allowed.

    Patches both the env var and the already-imported constant snapshot.
    Callers must hold their loader lock: os.environ is process-global.
    """
    if allow_download:
        yield
        return
    restore_env = os.environ.get("HF_HUB_OFFLINE")
    hf_const = None
    restore_const = None
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        try:
            import huggingface_hub.constants as hf_const
            restore_const = hf_const.HF_HUB_OFFLINE
            hf_const.HF_HUB_OFFLINE = True
        except ImportError:
            hf_const = None
        yield
    finally:
        if restore_env is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = restore_env
        if hf_const is not None:
            hf_const.HF_HUB_OFFLINE = restore_const


_erf = np.vectorize(math.erf, otypes=[np.float64])


def _gelu(x: np.ndarray) -> np.ndarray:
    """Exact (erf) GELU, matching torch.nn.GELU's default."""
    x = x.astype(np.float64)
    return x * 0.5 * (1.0 + _erf(x / math.sqrt(2.0)))


def _activation(name: str):
    n = (name or "").lower()
    if "gelu" in n:
        return _gelu
    if "tanh" in n:
        return np.tanh
    if "relu" in n:
        return lambda x: np.maximum(x, 0.0)
    if "sigmoid" in n:
        return lambda x: 1.0 / (1.0 + np.exp(-x))
    return lambda x: x  # Identity / unknown


def _read_json(path: Path) -> dict | list:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _tensor(tensors: dict[str, np.ndarray], suffix: str) -> np.ndarray | None:
    for k, v in tensors.items():
        if k == suffix or k.endswith("." + suffix):
            return v.astype(np.float64)
    return None


class OnnxCrossEncoder:
    """ONNX-backed cross-encoder returning raw logits."""

    def __init__(self, repo_id: str, *, onnx_file: str = "onnx/model.onnx",
                 max_length: int = 512, allow_download: bool = True):
        self.repo_id = repo_id
        self.max_length = max_length
        self._steps: list = []
        with _hf_offline(allow_download):
            root = self._fetch_files(repo_id, onnx_file)
        self._build(root, onnx_file)

    # -- loading ---------------------------------------------------------
    @staticmethod
    def _fetch_files(repo_id: str, onnx_file: str) -> dict[str, Path]:
        from huggingface_hub import hf_hub_download

        def get(name: str, optional: bool = False) -> Path | None:
            try:
                return Path(hf_hub_download(repo_id, name))
            except Exception:
                if optional:
                    return None
                raise

        files: dict[str, Path] = {
            onnx_file: get(onnx_file),
            "tokenizer.json": get("tokenizer.json"),
        }
        modules_path = get("modules.json", optional=True)
        if modules_path is not None:
            files["modules.json"] = modules_path
            for mod in _read_json(modules_path):
                sub = mod.get("path") or ""
                if not sub:
                    continue
                for fn in ("config.json", "model.safetensors"):
                    p = get(f"{sub}/{fn}", optional=True)
                    if p is not None:
                        files[f"{sub}/{fn}"] = p
        return files

    def _build(self, files: dict[str, Path], onnx_file: str) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(str(files["tokenizer.json"]))
        pad_id, pad_tok = 0, "[PAD]"
        for cand in ("[PAD]", "<pad>", "<|pad|>"):
            tid = tok.token_to_id(cand)
            if tid is not None:
                pad_id, pad_tok = tid, cand
                break
        tok.enable_truncation(max_length=self.max_length)
        tok.enable_padding(pad_id=pad_id, pad_token=pad_tok)
        self._tok = tok

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, min(4, os.cpu_count() or 1))
        self._sess = ort.InferenceSession(
            str(files[onnx_file]), sess_options=opts,
            providers=["CPUExecutionProvider"])
        self._input_names = [i.name for i in self._sess.get_inputs()]

        self._steps = []
        modules_path = files.get("modules.json")
        if modules_path is None:
            return
        from safetensors.numpy import load_file
        for mod in _read_json(modules_path):
            typ = str(mod.get("type", ""))
            sub = mod.get("path") or ""
            cfg_path = files.get(f"{sub}/config.json")
            cfg = _read_json(cfg_path) if cfg_path else {}
            st_path = files.get(f"{sub}/model.safetensors")
            tensors = load_file(str(st_path)) if st_path else {}
            if typ.endswith("Pooling"):
                mode = str(cfg.get("pooling_mode", "cls")).lower()
                if not cfg.get("pooling_mode") and cfg.get("pooling_mode_mean_tokens"):
                    mode = "mean"
                self._steps.append(("pool", mode))
            elif typ.endswith("Dense"):
                w = _tensor(tensors, "weight")
                if w is None:
                    raise ValueError(f"Dense module {sub!r} has no weight")
                self._steps.append(("dense", w, _tensor(tensors, "bias"),
                                    _activation(str(cfg.get("activation_function", "")))))
            elif typ.endswith("LayerNorm"):
                self._steps.append(("ln", _tensor(tensors, "weight"),
                                    _tensor(tensors, "bias"),
                                    float(cfg.get("eps", 1e-5))))
            # Transformer (the backbone) and unknown modules: nothing to do.

    # -- inference -------------------------------------------------------
    def _head(self, hidden: np.ndarray, mask: np.ndarray) -> np.ndarray:
        h = hidden.astype(np.float64)
        if h.ndim == 2:  # already [n, d] or [n, 1]/[n] logits: pass through
            return h[:, 0] if h.shape[1] == 1 else h
        if h.ndim == 3 and h.shape[-1] == 1 and not self._steps:
            return h[:, 0, 0]
        for step in self._steps:
            kind = step[0]
            if kind == "pool":
                if h.ndim != 3:
                    continue
                if step[1] == "mean":
                    m = mask[:, :, None].astype(np.float64)
                    h = (h * m).sum(axis=1) / np.maximum(m.sum(axis=1), 1e-9)
                else:
                    h = h[:, 0]
            elif kind == "dense":
                _, w, b, act = step
                h = h @ w.T
                if b is not None:
                    h = h + b
                h = act(h)
            elif kind == "ln":
                _, w, b, eps = step
                mu = h.mean(axis=-1, keepdims=True)
                var = ((h - mu) ** 2).mean(axis=-1, keepdims=True)
                h = (h - mu) / np.sqrt(var + eps)
                if w is not None:
                    h = h * w
                if b is not None:
                    h = h + b
        if h.ndim == 3:  # no pooling module: default to CLS
            h = h[:, 0]
        return h[:, 0] if h.ndim == 2 else h

    def score(self, query: str, texts: list[str], batch_size: int = 16) -> list[float]:
        out: list[float] = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            encs = self._tok.encode_batch([(query, t) for t in chunk])
            ids = np.asarray([e.ids for e in encs], dtype=np.int64)
            mask = np.asarray([e.attention_mask for e in encs], dtype=np.int64)
            feeds = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self._input_names:
                feeds["token_type_ids"] = np.asarray(
                    [e.type_ids for e in encs], dtype=np.int64)
            feeds = {k: v for k, v in feeds.items() if k in self._input_names}
            hidden = self._sess.run(None, feeds)[0]
            out.extend(float(x) for x in self._head(hidden, mask))
        return out
