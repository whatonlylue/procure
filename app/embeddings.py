"""Local embedding backend: potion-retrieval-32M via model2vec.

Static (non-transformer) embeddings distilled for retrieval: 512 dims,
MIT licensed. Inference is a numpy token-embedding lookup plus a weighted
mean -- there is no neural forward pass, so there is no GPU mode: ``device``
is always ``"cpu"`` and no provider selection exists. (Inside model2vec,
``cuda``/``device`` arguments appear only in the ``distill``/``train``
helpers, which procure never uses.)

Scratch benchmark on Apple Silicon (CPU): ~16k texts/s batched,
~1ms single-query latency, batch-of-32 in well under a millisecond.

Runtime deps (see pyproject): ``model2vec`` (which pulls ``tokenizers``
and ``safetensors``) plus ``huggingface_hub`` for the one-time model
download. No torch, no transformers, no onnxruntime.

Tuning (env flags, also mirrored on :class:`app.config.Settings`):

- ``PROCURE_EMBEDDING_MODEL`` -- HF repo (default
  ``minishlab/potion-retrieval-32M``).
- ``PROCURE_EMBEDDING_DIR`` -- use a pre-downloaded repo checkout instead
  of the HF cache (offline use).
- ``PROCURE_EMBEDDING_BATCH`` -- texts per inference step (default 32).
- ``PROCURE_EMBEDDING_MAX_LENGTH`` -- token cap per text (default 512,
  the model2vec default).
"""
from __future__ import annotations

import os
import threading

import numpy as np

MODEL_ID = "minishlab/potion-retrieval-32M"
DIM = 512
DEFAULT_MAX_LENGTH = 512

#: Stable id recorded next to stored vectors so a backend switch (or the
#: legacy hash corpus, which has no tag) is reported instead of silently
#: serving nonsense from dimension-compatible but meaningless vectors.
BACKEND = "potion-retrieval-32m"

#: model2vec inference is CPU-only by design (pure numpy); reported in
#: info()/health so callers can see there is no GPU to select.
DEVICE = "cpu"


class Model2VecEmbedder:
    """potion-retrieval-32M embeddings via model2vec, L2-normalized.

    Model download + object creation are lazy (first :meth:`embed` call)
    and thread-safe, so importing this module and constructing the
    embedder never touch the network.
    """

    dim = DIM
    backend = BACKEND
    device = DEVICE
    #: Weight of the dense channel in hybrid convex fusion. The static
    #: model is a weak ranker next to BM25, so it gets a small share.
    fusion_dense_weight = 0.3

    def __init__(self, model: str = MODEL_ID,
                 model_dir: str | None = None,
                 batch_size: int = 32,
                 max_length: int = DEFAULT_MAX_LENGTH) -> None:
        self.model = model or MODEL_ID
        self.model_dir = model_dir or None
        self.batch_size = max(1, int(batch_size))
        self.max_length = max(1, int(max_length))
        self._lock = threading.Lock()
        self._model = None  # model2vec.StaticModel, created lazily

    # -- introspection (never loads the model) --------------------------
    @property
    def loaded(self) -> bool:
        return self._model is not None

    def info(self) -> dict:
        """Cheap status for /api/health: no download, no inference."""
        return {
            "backend": self.backend,
            "model": self.model,
            "dim": self.dim,
            "device": self.device,
            "loaded": self.loaded,
        }

    # -- public API ------------------------------------------------------
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts to L2-normalized 512-dim vectors."""
        if not texts:
            return []
        self._ensure_loaded()
        assert self._model is not None
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            chunk = [t if isinstance(t, str) else "" for t in texts[i:i + self.batch_size]]
            out.extend(self._encode_batch(chunk))
        return out

    # -- lazy setup ------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            try:
                from model2vec import StaticModel
            except ImportError as e:
                raise RuntimeError(
                    "The embedding backend needs the 'model2vec' package "
                    "(pip install model2vec)."
                ) from e
            target = self._resolve_target()
            try:
                model = StaticModel.from_pretrained(target)
            except Exception as e:
                raise RuntimeError(
                    f"Could not load embedding model {target}: {e}. "
                    "Pass a local checkout via PROCURE_EMBEDDING_DIR "
                    "for offline use.") from e
            self._model = model
            # Warmup: validates the output width before any document
            # is ingested.
            width = self._encode_batch(["procure warmup"])
            if len(width[0]) != self.dim:
                raise RuntimeError(
                    f"Embedding model output width {len(width[0])} != "
                    f"expected {self.dim} ({target}).")

    def _resolve_target(self) -> str:
        """Return the HF repo id or local checkout path to load."""
        if self.model_dir is None:
            return self.model
        root = os.path.abspath(os.path.expanduser(self.model_dir))
        if not os.path.isdir(root):
            raise RuntimeError(
                f"PROCURE_EMBEDDING_DIR={root} does not exist; expected a "
                f"checkout of {self.model} (or unset it to download).")
        return root

    # -- inference -------------------------------------------------------
    def _encode_batch(self, texts: list[str]) -> list[list[float]]:
        assert self._model is not None
        vecs = np.asarray(
            self._model.encode(texts, show_progress_bar=False,
                               max_length=self.max_length),
            dtype=np.float32)
        # The model ships with normalization enabled, but re-normalize
        # defensively so the cosine-search contract holds no matter how
        # the model was distilled or loaded.
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (vecs / norms).astype(np.float32).tolist()


# ---------------------------------------------------------------------------
# Granite Embedding R2 small (int8 ONNX): optional "accurate" backend
# ---------------------------------------------------------------------------

GRANITE_MODEL_ID = "onnx-community/granite-embedding-small-english-r2-ONNX"
GRANITE_BACKEND = "granite-r2-small-int8"
GRANITE_ONNX_FILE = "onnx/model_quantized.onnx"
#: The ONNX export keeps weights in a sidecar file next to the graph.
GRANITE_ONNX_DATA = "onnx/model_quantized.onnx_data"
GRANITE_DIM = 384
GRANITE_MAX_LENGTH = 512
GRANITE_BATCH = 16
GRANITE_FUSION_WEIGHT = 0.7


def _stage_onnx(graph: str, data: str, name: str) -> str:
    """Return a path to ``graph`` whose external-data file sits beside it.

    The HF cache stores each file as a symlink into ``blobs/``; onnxruntime
    resolves symlinks and then rejects external data living outside the
    graph's (resolved) directory. Hardlink (or copy) both real files into
    one private directory next to the cache instead.
    """
    import shutil

    graph_real, data_real = os.path.realpath(graph), os.path.realpath(data)
    stage = os.path.join(os.path.dirname(os.path.dirname(graph_real)),
                         "procure-staged", name)
    os.makedirs(stage, exist_ok=True)
    for src, fn in ((graph_real, os.path.basename(graph)),
                    (data_real, os.path.basename(data))):
        dst = os.path.join(stage, fn)
        if os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src):
            continue
        tmp = dst + ".tmp"
        if os.path.exists(tmp):
            os.remove(tmp)
        try:
            os.link(src, tmp)
        except OSError:
            shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
    return os.path.join(stage, os.path.basename(graph))


class OnnxEmbedder:
    """Granite Embedding R2 small (int8 ONNX): CLS pooling, L2-normalized.

    Same interface as :class:`Model2VecEmbedder`. Pooling is CLS
    (``1_Pooling/config.json`` of ibm-granite/granite-embedding-small-english-r2
    sets only ``pooling_mode_cls_token``), there is no Normalize module so we
    normalize ourselves, and the model card documents no query/document
    prompt prefix. Loading is lazy and thread-safe; construction never
    touches the network.
    """

    dim = GRANITE_DIM
    backend = GRANITE_BACKEND
    device = DEVICE
    fusion_dense_weight = GRANITE_FUSION_WEIGHT

    def __init__(self, model: str = GRANITE_MODEL_ID,
                 model_dir: str | None = None,
                 batch_size: int = GRANITE_BATCH,
                 max_length: int = GRANITE_MAX_LENGTH,
                 allow_download: bool = True) -> None:
        self.model = model or GRANITE_MODEL_ID
        self.model_dir = model_dir or None
        self.batch_size = max(1, int(batch_size))
        self.max_length = max(2, min(int(max_length), GRANITE_MAX_LENGTH))
        self.allow_download = allow_download
        self._lock = threading.Lock()
        self._sess = None
        self._tok = None
        self._pad_id = 0
        self._input_names: list[str] = []

    @property
    def loaded(self) -> bool:
        return self._sess is not None

    def info(self) -> dict:
        return {"backend": self.backend, "model": self.model,
                "dim": self.dim, "device": self.device,
                "loaded": self.loaded}

    # -- public API ------------------------------------------------------
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts to L2-normalized float32 vectors (input order kept)."""
        if not texts:
            return []
        self._ensure_loaded()
        clean = [t if isinstance(t, str) else "" for t in texts]
        encs = self._tok.encode_batch(clean)
        # Length-sorted batches keep padding (and attention cost) minimal.
        order = sorted(range(len(clean)), key=lambda i: len(encs[i].ids))
        out: list[list[float] | None] = [None] * len(clean)
        for start in range(0, len(order), self.batch_size):
            idx = order[start:start + self.batch_size]
            vecs = self._run([encs[i] for i in idx])
            for i, v in zip(idx, vecs):
                out[i] = v
        return out  # type: ignore[return-value]

    # -- lazy setup ------------------------------------------------------
    def _locate(self) -> tuple[str, str]:
        """(tokenizer.json, staged model path); downloads when allowed."""
        if self.model_dir is not None:
            root = os.path.abspath(os.path.expanduser(self.model_dir))
            if not os.path.isdir(root):
                raise RuntimeError(
                    f"PROCURE_EMBEDDING_DIR={root} does not exist; expected "
                    f"a checkout of {self.model}.")
            return (os.path.join(root, "tokenizer.json"),
                    os.path.join(root, GRANITE_ONNX_FILE))
        from huggingface_hub import hf_hub_download

        from app.rerank_onnx import _hf_offline
        try:
            with _hf_offline(self.allow_download):
                tok = hf_hub_download(self.model, "tokenizer.json")
                graph = hf_hub_download(self.model, GRANITE_ONNX_FILE)
                data = hf_hub_download(self.model, GRANITE_ONNX_DATA)
        except Exception as e:
            raise RuntimeError(
                f"Could not fetch embedding model {self.model}: {e}. Set "
                "PROCURE_EMBEDDING_DIR to a local checkout for offline use."
            ) from e
        return tok, _stage_onnx(graph, data, "granite-r2-small-int8")

    def _ensure_loaded(self) -> None:
        if self._sess is not None:
            return
        with self._lock:
            if self._sess is not None:
                return
            import onnxruntime as ort
            from tokenizers import Tokenizer
            tok_path, graph = self._locate()
            tok = Tokenizer.from_file(tok_path)
            pad = tok.token_to_id("[PAD]")
            self._pad_id = 0 if pad is None else pad
            tok.enable_truncation(max_length=self.max_length)
            self._tok = tok
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = max(1, min(8, os.cpu_count() or 1))
            try:
                sess = ort.InferenceSession(
                    graph, sess_options=opts,
                    providers=["CPUExecutionProvider"])
            except Exception as e:
                raise RuntimeError(
                    f"Could not load embedding model {graph}: {e}") from e
            self._input_names = [i.name for i in sess.get_inputs()]
            self._sess = sess
            width = self._run([tok.encode("procure warmup")])
            if len(width[0]) != self.dim:
                self._sess = None
                raise RuntimeError(
                    f"Embedding model output width {len(width[0])} != "
                    f"expected {self.dim} ({self.model}).")

    # -- inference -------------------------------------------------------
    def _run(self, encs) -> list[list[float]]:
        assert self._sess is not None
        n = max(len(e.ids) for e in encs)
        ids = np.full((len(encs), n), self._pad_id, dtype=np.int64)
        mask = np.zeros((len(encs), n), dtype=np.int64)
        for r, e in enumerate(encs):
            ids[r, :len(e.ids)] = e.ids
            mask[r, :len(e.ids)] = 1
        feeds = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._input_names:
            feeds["token_type_ids"] = np.zeros_like(ids)
        feeds = {k: v for k, v in feeds.items() if k in self._input_names}
        hidden = np.asarray(self._sess.run(None, feeds)[0], dtype=np.float32)
        vecs = hidden[:, 0] if hidden.ndim == 3 else hidden  # CLS pooling
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (vecs / norms).astype(np.float32).tolist()


def build_embedder(settings):
    """Embedder for ``settings.embedding_backend`` (never loads a model)."""
    if getattr(settings, "embedding_backend", "potion") == "granite":
        return OnnxEmbedder(
            model=settings.granite_model,
            model_dir=settings.granite_dir or None,
            batch_size=GRANITE_BATCH,
            max_length=settings.embedding_max_length,
            allow_download=settings.embedding_allow_download)
    return Model2VecEmbedder(
        model=settings.embedding_model,
        model_dir=settings.embedding_dir or None,
        batch_size=settings.embedding_batch,
        max_length=settings.embedding_max_length)
