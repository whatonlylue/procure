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
