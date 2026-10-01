"""Local default vector store: SQLite + numpy cosine search. Zero extra deps."""
from __future__ import annotations

import contextlib
import os
import sqlite3
from collections.abc import Iterator

import numpy as np

from app.vectordb.base import ChunkHit, chunk_index

_IN_CHUNK = 500


def _batched(seq: list[str], n: int = _IN_CHUNK) -> Iterator[list[str]]:
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


class DimensionMismatchError(Exception):
    """Raised when an embedding batch mixes vector dimensions."""


class SqliteVectorStore:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        # Chunks skipped by the most recent search() because their stored
        # dimension differs from the query (embedding backend was switched).
        self.last_skipped = 0
        with self._session() as c:
            c.execute(
                """CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    embedding BLOB NOT NULL
                )"""
            )
            c.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id)")
            c.execute(
                """CREATE TABLE IF NOT EXISTS store_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL DEFAULT ''
                )"""
            )

    def _connect(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.path, timeout=10.0)
        c.execute("PRAGMA journal_mode=WAL").fetchone()
        c.execute("PRAGMA busy_timeout=5000")
        c.execute("PRAGMA synchronous=NORMAL")
        return c

    @contextlib.contextmanager
    def _session(self) -> Iterator[sqlite3.Connection]:
        c = self._connect()
        try:
            with c:
                yield c
        finally:
            c.close()

    def stored_dim(self) -> int | None:
        """Embedding dimension of the stored corpus, if any vectors exist."""
        with self._session() as c:
            row = c.execute(
                "SELECT value FROM store_meta WHERE key='dim'").fetchone()
            if row is not None:
                try:
                    return int(row[0])
                except ValueError:
                    return None
            blob = c.execute(
                "SELECT embedding FROM chunks LIMIT 1").fetchone()
        if blob is None:
            return None
        return len(np.frombuffer(blob[0], dtype=np.float32))

    def upsert(self, ids: list[str], doc_ids: list[str],
               embeddings: list[list[float]], texts: list[str]) -> None:
        dims = {len(e) for e in embeddings}
        if len(dims) > 1:
            raise DimensionMismatchError(
                f"Embedding batch mixes dimensions {sorted(dims)}")
        rows = [
            (cid, did, txt, np.asarray(emb, dtype=np.float32).tobytes())
            for cid, did, emb, txt in zip(ids, doc_ids, embeddings, texts)
        ]
        with self._session() as c:
            c.executemany(
                """INSERT INTO chunks (chunk_id, doc_id, text, embedding)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(chunk_id) DO UPDATE SET
                     doc_id=excluded.doc_id, text=excluded.text,
                     embedding=excluded.embedding""",
                rows,
            )
            if dims:
                c.execute(
                    "INSERT INTO store_meta (key, value) VALUES ('dim', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (str(next(iter(dims))),),
                )

    def delete_by_doc(self, doc_id: str) -> int:
        with self._session() as c:
            cur = c.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
        return cur.rowcount

    def search(self, query_embedding: list[float], top_k: int,
               doc_ids: list[str] | None = None) -> list[ChunkHit]:
        q = np.asarray(query_embedding, dtype=np.float32)
        qn = float(np.linalg.norm(q))
        if qn > 0:
            q = q / qn
        with self._session() as c:
            # Embeddings first, text later: blobs stack into one matrix for
            # a single matmul, and text is fetched for the winners only.
            if doc_ids:
                wanted = list(dict.fromkeys(doc_ids))
                rows: list[tuple] = []
                for batch in _batched(wanted):
                    ph = ",".join("?" for _ in batch)
                    rows.extend(c.execute(
                        f"SELECT chunk_id, doc_id, embedding FROM chunks "
                        f"WHERE doc_id IN ({ph})",
                        batch,
                    ).fetchall())
            else:
                rows = c.execute(
                    "SELECT chunk_id, doc_id, embedding FROM chunks").fetchall()
        ids: list[str] = []
        dids: list[str] = []
        vecs: list[np.ndarray] = []
        skipped = 0
        for cid, did, blob in rows:
            v = np.frombuffer(blob, dtype=np.float32)
            if v.shape != q.shape:
                skipped += 1
                continue
            ids.append(cid)
            dids.append(did)
            vecs.append(v)
        self.last_skipped = skipped
        if not vecs:
            return []
        mat = np.stack(vecs)
        norms = np.linalg.norm(mat, axis=1)
        norms[norms == 0] = 1.0
        scores = (mat @ q) / norms
        order = np.argsort(-scores, kind="stable")[:max(0, top_k)]
        winners = [(ids[i], dids[i], float(scores[i])) for i in order.tolist()]
        texts: dict[str, str] = {}
        with self._session() as c:
            for batch in _batched([w[0] for w in winners]):
                ph = ",".join("?" for _ in batch)
                for cid, text in c.execute(
                        f"SELECT chunk_id, text FROM chunks WHERE chunk_id IN ({ph})",
                        batch).fetchall():
                    texts[cid] = text
        return [ChunkHit(cid, did, texts.get(cid, ""), s)
                for cid, did, s in winners]

    def get_by_doc(self, doc_id: str) -> list[tuple[str, str]]:
        """(chunk_id, text) rows for one document, in chunk order."""
        with self._session() as c:
            rows = c.execute(
                "SELECT chunk_id, text FROM chunks WHERE doc_id=?", (doc_id,)
            ).fetchall()
        return sorted(rows, key=lambda r: chunk_index(r[0]))

    def count(self) -> int:
        with self._session() as c:
            return int(c.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

    def all_chunks(self) -> list[tuple[str, str, str]]:
        """All (chunk_id, doc_id, text) rows; used to backfill sparse stats."""
        with self._session() as c:
            return [(r[0], r[1], r[2]) for r in
                    c.execute("SELECT chunk_id, doc_id, text FROM chunks").fetchall()]
