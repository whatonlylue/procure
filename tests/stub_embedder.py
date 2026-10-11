"""Deterministic offline embedder for the test suite (not a product backend).

The production backend (:class:`app.embeddings.Model2VecEmbedder`)
downloads a ~120MB model on first use; tests inject this tiny stub instead
so the suite stays fast and offline. Vectors are deterministic per text
(seeded by SHA-256), L2-normalized, and 512-wide like the real backend.
Dense scores from the stub are meaningless noise by design -- the suite's
semantic assertions rest on the sparse BM25 channel and the heuristic
rerank, which is exactly what they exercise. Real embedding semantics are
covered by ``test_embeddings.py`` (fake-model unit tests plus a
live-model test).
"""
from __future__ import annotations

import hashlib
import os

import numpy as np

# Hermetic suite: no test may reach the network. Every service test imports
# this module, so pin the env-driven defaults that would otherwise start a
# background Hugging Face download (reranker) or load the Granite embedder.
os.environ["PROCURE_CROSS_ENCODER"] = ""
os.environ["PROCURE_CROSS_ENCODER_ALLOW_DOWNLOAD"] = "0"
os.environ["PROCURE_EMBEDDING_BACKEND"] = "potion"
os.environ["PROCURE_EMBEDDING_ALLOW_DOWNLOAD"] = "0"


class StubEmbedder:
    dim = 512
    backend = "test-stub"
    device = "cpu"

    def __init__(self, backend: str = "test-stub") -> None:
        self.backend = backend

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            seed = int.from_bytes(
                hashlib.sha256((text or "").encode("utf-8")).digest()[:4],
                "little")
            vec = np.random.RandomState(seed).randn(self.dim)
            norm = float(np.linalg.norm(vec)) or 1.0
            out.append((vec / norm).astype(np.float32).tolist())
        return out

    def info(self) -> dict:
        return {"backend": self.backend, "model": "test-stub",
                "dim": self.dim, "device": self.device,
                "loaded": True}
