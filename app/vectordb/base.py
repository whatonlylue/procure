"""Shared row types for the local sqlite vector store."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ChunkHit:
    chunk_id: str
    doc_id: str
    text: str
    score: float


def chunk_index(chunk_id: str) -> int:
    """Numeric suffix of a `{doc_id}:{i}` chunk id; unknown shapes sort last."""
    try:
        return int(chunk_id.rsplit(":", 1)[1])
    except (IndexError, ValueError):
        return 1 << 30

