"""Local default vector store: SQLite + numpy cosine search. Zero extra deps."""
from __future__ import annotations

import os
import sqlite3

import numpy as np

from app.vectordb.base import ChunkHit


def _chunk_index(chunk_id: str) -> int:
    """Numeric suffix of a `{doc_id}:{i}` chunk id; unknown shapes sort last."""
    try:
        return int(chunk_id.rsplit(":", 1)[1])
    except (IndexError, ValueError):
        return 1 << 30


class SqliteVectorStore:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        with self._connect() as c:
            c.execute(
                """CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    embedding BLOB NOT NULL
                )"""
            )
            c.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id)")

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def upsert(self, ids: list[str], doc_ids: list[str],
               embeddings: list[list[float]], texts: list[str]) -> None:
        rows = [
            (cid, did, txt, np.asarray(emb, dtype=np.float32).tobytes())
            for cid, did, emb, txt in zip(ids, doc_ids, embeddings, texts)
        ]
        with self._connect() as c:
            c.executemany(
                """INSERT INTO chunks (chunk_id, doc_id, text, embedding)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(chunk_id) DO UPDATE SET
                     doc_id=excluded.doc_id, text=excluded.text,
                     embedding=excluded.embedding""",
                rows,
            )

    def delete_by_doc(self, doc_id: str) -> int:
        with self._connect() as c:
            cur = c.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
        return cur.rowcount

    def search(self, query_embedding: list[float], top_k: int,
               doc_ids: list[str] | None = None) -> list[ChunkHit]:
        q = np.asarray(query_embedding, dtype=np.float32)
        with self._connect() as c:
            if doc_ids:
                ph = ",".join("?" for _ in doc_ids)
                rows = c.execute(
                    f"SELECT chunk_id, doc_id, text, embedding FROM chunks WHERE doc_id IN ({ph})",
                    doc_ids,
                ).fetchall()
            else:
                rows = c.execute("SELECT chunk_id, doc_id, text, embedding FROM chunks").fetchall()
        hits: list[ChunkHit] = []
        for cid, did, text, blob in rows:
            v = np.frombuffer(blob, dtype=np.float32)
            if v.shape != q.shape:
                continue
            denom = float(np.linalg.norm(v) * np.linalg.norm(q))
            score = float(np.dot(v, q) / denom) if denom > 0 else 0.0
            hits.append(ChunkHit(cid, did, text, score))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]

    def get_by_doc(self, doc_id: str) -> list[tuple[str, str]]:
        """(chunk_id, text) rows for one document, in chunk order."""
        with self._connect() as c:
            rows = c.execute(
                "SELECT chunk_id, text FROM chunks WHERE doc_id=?", (doc_id,)
            ).fetchall()
        return sorted(rows, key=lambda r: _chunk_index(r[0]))

    def count(self) -> int:
        with self._connect() as c:
            return int(c.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

    def all_chunks(self) -> list[tuple[str, str, str]]:
        """All (chunk_id, doc_id, text) rows; used to backfill sparse stats."""
        with self._connect() as c:
            return [(r[0], r[1], r[2]) for r in
                    c.execute("SELECT chunk_id, doc_id, text FROM chunks").fetchall()]
