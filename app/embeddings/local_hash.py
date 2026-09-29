"""Local default embedder: hashed word n-gram vectors. Zero dependencies."""
from __future__ import annotations

import hashlib
import math
import re

_WORD = re.compile(r"[a-z0-9]+")


class HashEmbedder:
    """Fast local bag-of-words hash embedding, L2-normalized for cosine search."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        toks = _WORD.findall(text.lower())
        feats = toks + [f"{a} {b}" for a, b in zip(toks, toks[1:])]
        for f in feats:
            h = int(hashlib.md5(f.encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec
