"""Hybrid search: FTS5 sparse store, convex score fusion, cross-encoder reranking.

Sparse scoring is dependency-free (stdlib + sqlite3 FTS5). The cross-encoder
rerank path loads lazily and offline-first; when the model is unavailable
scoring falls back to the coverage heuristic.
"""
from __future__ import annotations

import contextlib
import logging
import math
import os
import re
import sqlite3
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass

from app.store import keepalive

_log = logging.getLogger(__name__)

_WORD = re.compile(r"[a-z0-9]+")
RRF_K = 60
# Weight of the dense list in convex fusion (potion's dense list is the
# weaker one; ~0.7 suits a contextual embedder).
DENSE_WEIGHT = 0.3

_IN_CHUNK = 500


def _batched(seq: list[str], n: int = _IN_CHUNK) -> Iterator[list[str]]:
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


# Minimal stopword list so coverage scoring is driven by content terms.
_STOPWORDS = frozenset(
    "a an the and or but if then else when what which who whom whose why how "
    "is are was were be been being do does did done have has had having will would "
    "can could shall should may might must of in on at to for with by from as into "
    "it its this that these those there their them they he she we you i me my our "
    "your his her us about over under again once here".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens (shared by BM25 and coverage)."""
    return _WORD.findall((text or "").lower())


def content_terms(query: str) -> list[str]:
    """Deduped query terms with stopwords removed (falls back to all terms)."""
    seen: list[str] = []
    for t in tokenize(query):
        if t not in _STOPWORDS and t not in seen:
            seen.append(t)
    if not seen:
        for t in tokenize(query):
            if t not in seen:
                seen.append(t)
    return seen


@dataclass
class SparseHit:
    chunk_id: str
    doc_id: str
    text: str
    score: float


def _fts_phrase(term: str) -> str:
    """One FTS5 MATCH phrase: double-quoted with embedded quotes doubled."""
    return '"' + term.replace('"', '""') + '"'


def _fts_match(terms: list[str]) -> str:
    """OR of FTS5 phrases (callers pass ``content_terms`` output).

    Terms come from :func:`tokenize` (``[a-z0-9]+``), so they carry no FTS5
    syntax of their own; quoting still guards the general case.
    """
    return " OR ".join(_fts_phrase(t) for t in terms)


class SparseStore:
    """BM25 sparse retrieval backed by SQLite FTS5 (stdlib sqlite3, no deps).

    One FTS5 table holds ``(chunk_id, doc_id, text)``; ranking uses the
    built-in ``bm25()`` auxiliary (negated to a positive score, larger is
    better). Ingest/delete/reingest stay incremental with no global rebuild.

    Databases written by the previous hand-rolled ``chunks``/``postings``
    schema are migrated once on open (rows copied into FTS5, old tables
    dropped); the vector store can additionally backfill an empty sparse
    store (see ``RAGService._backfill_sparse``).
    """

    _TABLE = "chunks_fts"

    _MAP = "fts_rowmap"
    _TOKENIZER = "porter unicode61 remove_diacritics 2"

    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        with self._session() as c:
            c.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS {self._TABLE} USING fts5("
                "chunk_id UNINDEXED, doc_id UNINDEXED, text, "
                f"tokenize='{self._TOKENIZER}')"
            )
            c.execute(
                f"CREATE TABLE IF NOT EXISTS {self._MAP} ("
                "chunk_id TEXT PRIMARY KEY, doc_id TEXT NOT NULL, "
                "fts_rowid INTEGER NOT NULL)"
            )
            c.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{self._MAP}_doc "
                f"ON {self._MAP}(doc_id)"
            )
            self._migrate_legacy(c)
        self._upgrade_tokenizer()
        self._backfill_rowmap()
        keepalive(self.path)

    def _fts_sql(self, c: sqlite3.Connection) -> str:
        row = c.execute(
            "SELECT sql FROM sqlite_master WHERE name=?", (self._TABLE,)
        ).fetchone()
        return (row[0] if row else "") or ""

    def _upgrade_tokenizer(self) -> None:
        """Rebuild an FTS table created with the pre-stemming tokenizer.

        Rowids are preserved so ``fts_rowmap`` stays valid. Runs in one
        write transaction and re-checks after taking the lock, so two
        processes opening the same DB cannot both rebuild.
        """
        with self._session() as c:
            if "porter" in self._fts_sql(c):
                return
            c.execute("BEGIN IMMEDIATE")
            if "porter" in self._fts_sql(c):
                return
            new = f"{self._TABLE}_new"
            c.execute(f"DROP TABLE IF EXISTS {new}")
            c.execute(
                f"CREATE VIRTUAL TABLE {new} USING fts5("
                "chunk_id UNINDEXED, doc_id UNINDEXED, text, "
                f"tokenize='{self._TOKENIZER}')"
            )
            c.execute(
                f"INSERT INTO {new}(rowid, chunk_id, doc_id, text) "
                f"SELECT rowid, chunk_id, doc_id, text FROM {self._TABLE}"
            )
            c.execute(f"DROP TABLE {self._TABLE}")
            c.execute(f"ALTER TABLE {new} RENAME TO {self._TABLE}")

    def _backfill_rowmap(self) -> None:
        """Populate ``fts_rowmap`` once for an FTS table that predates it."""
        with self._session() as c:
            if c.execute(f"SELECT 1 FROM {self._MAP} LIMIT 1").fetchone():
                return
            if not c.execute(f"SELECT 1 FROM {self._TABLE} LIMIT 1").fetchone():
                return
            c.execute("BEGIN IMMEDIATE")
            if c.execute(f"SELECT 1 FROM {self._MAP} LIMIT 1").fetchone():
                return
            # Old data may hold duplicate chunk_ids: keep the newest row.
            c.execute(
                f"DELETE FROM {self._TABLE} WHERE rowid NOT IN "
                f"(SELECT MAX(rowid) FROM {self._TABLE} GROUP BY chunk_id)"
            )
            c.execute(
                f"INSERT INTO {self._MAP}(chunk_id, doc_id, fts_rowid) "
                f"SELECT chunk_id, doc_id, rowid FROM {self._TABLE}"
            )

    def _migrate_legacy(self, c: sqlite3.Connection) -> None:
        """Copy hand-rolled ``chunks``/``postings`` rows into FTS5, then drop."""
        tables = {r[0] for r in c.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
        ).fetchall()}
        if "chunks" not in tables:
            return
        count = int(c.execute(f"SELECT COUNT(*) FROM {self._TABLE}").fetchone()[0])
        if count == 0:
            c.execute(
                f"INSERT INTO {self._TABLE} (chunk_id, doc_id, text) "
                "SELECT chunk_id, doc_id, text FROM chunks"
            )
        c.execute("DROP TABLE IF EXISTS postings")
        c.execute("DROP TABLE IF EXISTS chunks")
        c.execute("DROP INDEX IF EXISTS idx_post_term")
        c.execute("DROP INDEX IF EXISTS idx_chunks_doc")

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

    def upsert_chunks(self, ids: list[str], doc_ids: list[str], texts: list[str]) -> None:
        with self._session() as c:
            for cid, did, text in zip(ids, doc_ids, texts):
                old = c.execute(
                    f"SELECT fts_rowid FROM {self._MAP} WHERE chunk_id=?", (cid,)
                ).fetchone()
                if old:
                    c.execute(f"DELETE FROM {self._TABLE} WHERE rowid=?", (old[0],))
                cur = c.execute(
                    f"INSERT INTO {self._TABLE} (chunk_id, doc_id, text)"
                    " VALUES (?, ?, ?)",
                    (cid, did, text),
                )
                c.execute(
                    f"INSERT OR REPLACE INTO {self._MAP}"
                    "(chunk_id, doc_id, fts_rowid) VALUES (?, ?, ?)",
                    (cid, did, cur.lastrowid),
                )

    def delete_by_doc(self, doc_id: str) -> int:
        with self._session() as c:
            rowids = [r[0] for r in c.execute(
                f"SELECT fts_rowid FROM {self._MAP} WHERE doc_id=?", (doc_id,)
            ).fetchall()]
            deleted = 0
            for batch in _batched(rowids):
                cur = c.executemany(
                    f"DELETE FROM {self._TABLE} WHERE rowid=?",
                    [(r,) for r in batch],
                )
                deleted += int(cur.rowcount or 0)
            c.execute(f"DELETE FROM {self._MAP} WHERE doc_id=?", (doc_id,))
            return deleted

    def count(self) -> int:
        with self._session() as c:
            return int(c.execute(f"SELECT COUNT(*) FROM {self._TABLE}").fetchone()[0])

    def idf_map(self, terms: list[str]) -> dict[str, float]:
        """IDF per term: ln(1 + (N - df + 0.5) / (df + 0.5)).

        Document frequency is counted through FTS5 MATCH, so it agrees with
        the retrieval index; the formula matches the previous hand-rolled
        store (used by the coverage fallback in ``RAGService.search``).
        """
        terms = list(dict.fromkeys(terms))
        if not terms:
            return {}
        with self._session() as c:
            n = int(c.execute(f"SELECT COUNT(*) FROM {self._TABLE}").fetchone()[0])
            if n == 0:
                return {}
            df = {
                t: int(c.execute(
                    f"SELECT COUNT(*) FROM {self._TABLE} "
                    f"WHERE {self._TABLE} MATCH ?", (_fts_phrase(t),),
                ).fetchone()[0])
                for t in terms
            }
        return {t: math.log(1.0 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def search(self, query: str, top_n: int,
               doc_ids: list[str] | None = None) -> list[SparseHit]:
        terms = content_terms(query)
        if not terms:
            return []
        match = _fts_match(terms)
        with self._session() as c:
            # bm25() ranks best-first with ORDER BY ascending (most negative
            # first); scores are negated so larger SparseHit.score is better.
            scored: dict[str, tuple[str, str, float]] = {}
            scopes = [None] if not doc_ids else list(_batched(list(doc_ids)))
            for scope in scopes:
                if scope is None:
                    rows = c.execute(
                        f"SELECT chunk_id, doc_id, text, bm25({self._TABLE}) AS r "
                        f"FROM {self._TABLE} WHERE {self._TABLE} MATCH ? "
                        "ORDER BY r",
                        (match,),
                    ).fetchall()
                else:
                    dph = ",".join("?" for _ in scope)
                    rows = c.execute(
                        f"SELECT chunk_id, doc_id, text, bm25({self._TABLE}) AS r "
                        f"FROM {self._TABLE} WHERE {self._TABLE} MATCH ? "
                        f"AND doc_id IN ({dph}) ORDER BY r",
                        [match, *scope],
                    ).fetchall()
                for cid, did, text, rank in rows:
                    score = -float(rank)
                    if cid not in scored or score > scored[cid][2]:
                        scored[cid] = (did, text, score)
            ranked = sorted(scored.items(), key=lambda kv: kv[1][2],
                            reverse=True)[:top_n]
        return [SparseHit(cid, did, text, score)
                for cid, (did, text, score) in ranked]


def rrf_fuse(ranked_lists: list[list[str]], k: int = RRF_K) -> dict[str, float]:
    """Reciprocal-rank fusion: sum over lists of 1 / (k + rank)."""
    fused: dict[str, float] = {}
    for ids in ranked_lists:
        for rank, cid in enumerate(ids, start=1):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank)
    return fused


def convex_fuse(dense: dict[str, float], sparse: dict[str, float],
                w_dense: float = DENSE_WEIGHT) -> dict[str, float]:
    """Min-max normalize each list, then ``w_dense*dense + (1-w_dense)*sparse``;
    a chunk missing from a list contributes 0 for it."""
    d = minmax_norm(dense)
    sp = minmax_norm(sparse)
    return {cid: w_dense * d.get(cid, 0.0) + (1.0 - w_dense) * sp.get(cid, 0.0)
            for cid in dict.fromkeys([*dense, *sparse])}


def minmax_norm(scores: dict[str, float]) -> dict[str, float]:
    """Min-max normalize to [0, 1]; all-equal maps to 1.0."""
    if not scores:
        return {}
    lo = min(scores.values())
    hi = max(scores.values())
    if hi <= lo:
        return {cid: 1.0 for cid in scores}
    span = hi - lo
    return {cid: (s - lo) / span for cid, s in scores.items()}


#: After a failed load the scorer stays on the coverage fallback for this
#: long, then retries (a transient network failure must not disable the
#: reranker until restart).
CROSS_ENCODER_RETRY_SECONDS = 600.0


class CrossEncoderScorer:
    """Local cross-encoder reranker (ONNX + numpy, no torch), loaded lazily.

    The first :meth:`score` call starts a background load (download + session
    creation) and returns None; callers fall back to the coverage heuristic
    until the model is ready. A failed load is logged once and retried after
    :data:`CROSS_ENCODER_RETRY_SECONDS`.
    :meth:`warmup` loads synchronously.
    """

    def __init__(self, model_name: str = "", allow_download: bool = True):
        self.model_name = (model_name or "").strip()
        self.allow_download = allow_download
        self._model = None
        self._tried = False
        self._failed_at: float | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        """True once a model is in memory (never triggers a load)."""
        return self.loaded

    @property
    def loaded(self) -> bool:
        with self._lock:
            return self._model is not None

    def _do_load(self) -> None:
        model = None
        try:
            from app.rerank_onnx import OnnxCrossEncoder
            model = OnnxCrossEncoder(self.model_name,
                                     allow_download=self.allow_download)
        except Exception as exc:  # offline/no cache, missing runtime, bad files
            _log.warning("cross-encoder %s unavailable, using coverage "
                         "fallback: %s", self.model_name, exc)
        with self._lock:
            self._model = model
            self._tried = True
            self._failed_at = None if model is not None else time.monotonic()
            self._thread = None

    def _retry_due(self) -> bool:
        """Caller holds the lock: a failed load older than the cooldown."""
        return (self._model is None and self._failed_at is not None
                and time.monotonic() - self._failed_at
                >= CROSS_ENCODER_RETRY_SECONDS)

    def warmup(self, block: bool = True) -> bool:
        """Load the model; returns True when it is ready."""
        if not self.model_name:
            return False
        with self._lock:
            if self._tried and self._retry_due():
                self._tried = False
                self._failed_at = None
            if self._tried:
                return self._model is not None
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._do_load, name="cross-encoder-load", daemon=True)
                self._thread.start()
            thread = self._thread
        if block:
            thread.join()
        return self.loaded

    def score(self, query: str, texts: list[str]) -> list[float] | None:
        """Cross-encoder relevance logits, or None when unavailable/loading."""
        if not self.model_name or not texts:
            return None
        with self._lock:
            model = self._model
            tried = self._tried and not self._retry_due()
        if model is None:
            if not tried:
                self.warmup(block=False)
            return None
        try:
            return [float(x) for x in model.score(query, texts)]
        except Exception:
            return None


def combine_answerability(cross: list[float] | None,
                          coverage: list[float]) -> list[float]:
    """Cross-encoder logits min-max scaled over the candidate pool;
    coverage alone when no cross-encoder scores are available.

    A sigmoid on raw logits saturated (Ettin emits roughly -2..+9, so every
    plausible candidate landed near 1.0 and the reranker added nothing).
    Min-max within the 15-30 candidate pool keeps the full spread; the
    all-equal case maps to 1.0 (see :func:`minmax_norm`).
    """
    if cross is None:
        return list(coverage)
    norm = minmax_norm({str(i): x for i, x in enumerate(cross)})
    return [norm[str(i)] for i in range(len(cross))]


def coverage_score(query: str, text: str, idf: dict[str, float] | None = None) -> float:
    """Lightweight answerability heuristic: IDF-weighted query-term coverage
    of the chunk plus a query-bigram phrase bonus. Returns [0, 1]."""
    q = content_terms(query)
    if not q:
        return 0.0
    c = tokenize(text)
    if not c:
        return 0.0
    cset = set(c)
    weights = [(idf.get(t, 1.0) if idf else 1.0) for t in q]
    hits = [w for t, w in zip(q, weights) if t in cset]
    base = sum(hits) / sum(weights) if sum(weights) > 0 else 0.0
    bonus = 0.0
    # Phrase bonus compares RAW query bigrams (stopwords kept) against raw
    # chunk bigrams: with stripped terms, "capital of france" would look for
    # a (capital, france) bigram that never occurs.
    raw = tokenize(query)
    if len(raw) > 1:
        qb = set(zip(raw, raw[1:]))
        cb = set(zip(c, c[1:]))
        bonus = 0.25 * len(qb & cb) / len(qb)
    return min(1.0, base + bonus)
