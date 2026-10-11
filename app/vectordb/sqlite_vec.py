"""Local default vector store: SQLite + numpy cosine search. Zero extra deps."""
from __future__ import annotations

import contextlib
import os
import sqlite3
import threading
from collections.abc import Iterator

import numpy as np

from app.store import keepalive
from app.vectordb.base import ChunkHit, chunk_index

_IN_CHUNK = 500


def _batched(seq: list[str], n: int = _IN_CHUNK) -> Iterator[list[str]]:
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


class DimensionMismatchError(Exception):
    """Raised when an embedding batch mixes vector dimensions."""


class _MatrixCache:
    """Row-normalized float32 matrix of every stored vector, in SELECT order."""

    def __init__(self, rows: list[tuple[str, str, bytes]], version: int):
        self.version = version
        self.ids = [r[0] for r in rows]
        self.doc_ids = [r[1] for r in rows]
        widths = {len(r[2]) for r in rows}
        # Mixed widths (backend switched without re-ingest): no matrix, the
        # store falls back to the per-query path that counts skipped rows.
        self.uniform = len(widths) == 1
        self.dim = next(iter(widths)) // 4 if self.uniform else 0
        self.matrix: np.ndarray | None = None
        self.by_doc: dict[str, list[int]] = {}
        if not self.uniform:
            return
        mat = np.frombuffer(b"".join(r[2] for r in rows),
                            dtype=np.float32).reshape(len(rows), self.dim)
        norms = np.linalg.norm(mat, axis=1)
        norms[norms == 0] = 1.0
        self.matrix = mat / norms[:, None]
        for i, did in enumerate(self.doc_ids):
            self.by_doc.setdefault(did, []).append(i)

    def select(self, doc_ids: list[str] | None) -> np.ndarray | None:
        """Row indices to score, or None for all rows."""
        if not doc_ids:
            return None
        rows: list[int] = []
        for did in dict.fromkeys(doc_ids):
            rows.extend(self.by_doc.get(did, ()))
        return np.asarray(rows, dtype=np.intp)


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
        keepalive(self.path)
        # Long-lived connection used only for PRAGMA data_version, which is
        # comparable across time on one connection (it ignores that
        # connection's own commits; writes here use per-call sessions).
        self._vconn: sqlite3.Connection | None = sqlite3.connect(
            self.path, timeout=10.0, check_same_thread=False)
        self._vlock = threading.Lock()
        self._cache: _MatrixCache | None = None
        self._dirty = True
        self._cache_lock = threading.Lock()

    def close(self) -> None:
        """Close the data_version connection; safe to call twice."""
        with self._vlock:
            c, self._vconn = self._vconn, None
        if c is not None:
            c.close()

    def _data_version(self) -> int | None:
        with self._vlock:
            if self._vconn is None:
                return None
            return int(self._vconn.execute("PRAGMA data_version").fetchone()[0])

    def _current_cache(self) -> _MatrixCache:
        """Cache matching the DB now, rebuilding lazily after any change."""
        with self._cache_lock:
            version = self._data_version()
            cache = self._cache
            if (cache is not None and not self._dirty and version is not None
                    and cache.version == version):
                return cache
            # Version is read before the SELECT, so a write racing the build
            # leaves an older version and forces a rebuild on the next query.
            self._dirty = False
            with self._session() as c:
                rows = c.execute(
                    "SELECT chunk_id, doc_id, embedding FROM chunks").fetchall()
            cache = _MatrixCache(rows, -1 if version is None else version)
            self._cache = cache
            return cache

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

    def stored_backend(self) -> str | None:
        """Embedding backend tag of the stored corpus, if any was recorded.

        Corpora written before backend tags existed (the hash era) report
        None so callers treat them as stale instead of comparable.
        """
        with self._session() as c:
            row = c.execute(
                "SELECT value FROM store_meta WHERE key='backend'").fetchone()
        return row[0] if row is not None else None

    def upsert(self, ids: list[str], doc_ids: list[str],
               embeddings: list[list[float]], texts: list[str],
               backend: str | None = None) -> None:
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
            if backend is not None:
                c.execute(
                    "INSERT INTO store_meta (key, value) VALUES ('backend', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (backend,),
                )
        self._dirty = True

    def replace_embeddings(self, ids: list[str], embeddings: list[list[float]]) -> int:
        """Overwrite the vectors of EXISTING chunks only (re-embed path).

        Never inserts (a chunk deleted meanwhile stays deleted) and leaves
        the recorded dim/backend alone: call :meth:`set_corpus_meta` once the
        whole corpus has been rewritten.
        """
        rows = [(np.asarray(e, dtype=np.float32).tobytes(), cid)
                for cid, e in zip(ids, embeddings)]
        with self._session() as c:
            n = sum(c.execute(
                "UPDATE chunks SET embedding=? WHERE chunk_id=?", r).rowcount
                for r in rows)
        self._dirty = True
        return n

    def set_corpus_meta(self, dim: int, backend: str | None) -> None:
        """Record the corpus dim/backend (after a complete re-embed)."""
        with self._session() as c:
            for key, val in (("dim", str(int(dim))), ("backend", backend)):
                if val is None:
                    continue
                c.execute(
                    "INSERT INTO store_meta (key, value) VALUES (?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (key, val))

    def delete_by_doc(self, doc_id: str) -> int:
        with self._session() as c:
            cur = c.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
        self._dirty = True
        return cur.rowcount

    @staticmethod
    def _unit(query_embedding: list[float]) -> np.ndarray:
        q = np.asarray(query_embedding, dtype=np.float32)
        qn = float(np.linalg.norm(q))
        return q / qn if qn > 0 else q

    def search(self, query_embedding: list[float], top_k: int,
               doc_ids: list[str] | None = None) -> list[ChunkHit]:
        q = self._unit(query_embedding)
        cache = self._current_cache()
        if not cache.uniform or cache.dim != q.shape[0]:
            return self._search_uncached(query_embedding, top_k, doc_ids)
        self.last_skipped = 0
        assert cache.matrix is not None
        sel = cache.select(doc_ids)
        if sel is None:
            scores = cache.matrix @ q
        elif sel.size == 0:
            return []
        else:
            scores = cache.matrix[sel] @ q
        order = np.argsort(-scores, kind="stable")[:max(0, top_k)]
        winners = []
        for i in order.tolist():
            row = i if sel is None else int(sel[i])
            winners.append((cache.ids[row], cache.doc_ids[row], float(scores[i])))
        return self._hits(winners)

    def _search_uncached(self, query_embedding: list[float], top_k: int,
                         doc_ids: list[str] | None = None) -> list[ChunkHit]:
        """Per-query read of every blob; handles mixed stored widths."""
        q = self._unit(query_embedding)
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
        return self._hits(winners)

    def _hits(self, winners: list[tuple[str, str, float]]) -> list[ChunkHit]:
        """Attach chunk text, fetched for the winners only."""
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
