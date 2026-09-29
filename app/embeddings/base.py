"""Embedder protocol: every backend (local hash, sbert, cloud) implements this."""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts -> L2-normalized vectors of length dim."""
        ...
