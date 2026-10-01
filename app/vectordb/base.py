"""VectorStore protocol shared by local sqlite and chroma backends."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


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


@runtime_checkable
class VectorStore(Protocol):
    def upsert(self, ids: list[str], doc_ids: list[str],
               embeddings: list[list[float]], texts: list[str]) -> None: ...
    def delete_by_doc(self, doc_id: str) -> int: ...
    def search(self, query_embedding: list[float], top_k: int,
               doc_ids: list[str] | None = None) -> list[ChunkHit]: ...
    def get_by_doc(self, doc_id: str) -> list[tuple[str, str]]: ...
    def count(self) -> int: ...
    def all_chunks(self) -> list[tuple[str, str, str]]:
        """All (chunk_id, doc_id, text) rows; used to backfill sparse stats."""
        ...
