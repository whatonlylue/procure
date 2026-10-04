"""Hybrid search: FTS5 sparse store, RRF fusion, CLEAR-style answerability.

Sparse scoring is dependency-free (stdlib + sqlite3 FTS5). Neural rerank paths
load lazily and offline-first; when a model is unavailable scoring falls
back to the coverage heuristic. See arXiv:2609.03482 (CLEAR) for the
``sigmoid(relevance) + alpha * entailment`` inference formula.
"""
from __future__ import annotations

import contextlib
import math
import os
import re
import sqlite3
import threading
from collections.abc import Iterator
from dataclasses import dataclass

_WORD = re.compile(r"[a-z0-9]+")
RRF_K = 60

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

    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        with self._session() as c:
            c.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS {self._TABLE} USING fts5("
                "chunk_id UNINDEXED, doc_id UNINDEXED, text, "
                "tokenize='unicode61 remove_diacritics 2')"
            )
            self._migrate_legacy(c)

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
                c.execute(f"DELETE FROM {self._TABLE} WHERE chunk_id=?", (cid,))
                c.execute(
                    f"INSERT INTO {self._TABLE} (chunk_id, doc_id, text)"
                    " VALUES (?, ?, ?)",
                    (cid, did, text),
                )

    def delete_by_doc(self, doc_id: str) -> int:
        with self._session() as c:
            cur = c.execute(f"DELETE FROM {self._TABLE} WHERE doc_id=?", (doc_id,))
            return int(cur.rowcount or 0)

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


@contextlib.contextmanager
def _hf_offline(allow_download: bool) -> Iterator[None]:
    """Force HuggingFace Hub offline unless downloads are allowed.

    Patches both the env var and the already-imported constant snapshot.
    Callers must hold their loader lock: os.environ is process-global.
    """
    if allow_download:
        yield
        return
    restore_env = os.environ.get("HF_HUB_OFFLINE")
    hf_const = None
    restore_const = None
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        try:
            import huggingface_hub.constants as hf_const
            restore_const = hf_const.HF_HUB_OFFLINE
            hf_const.HF_HUB_OFFLINE = True
        except ImportError:
            hf_const = None
        yield
    finally:
        if restore_env is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = restore_env
        if hf_const is not None:
            hf_const.HF_HUB_OFFLINE = restore_const


class CrossEncoderScorer:
    """Local cross-encoder reranker, loaded lazily and offline-first.

    Uses a cached sentence-transformers CrossEncoder model when available;
    never downloads unless ``allow_download`` is set. When the model cannot
    be loaded (not cached, library missing), :meth:`score` returns None and
    callers fall back to the coverage heuristic.
    """

    def __init__(self, model_name: str = "", allow_download: bool = False):
        self.model_name = (model_name or "").strip()
        self.allow_download = allow_download
        self._model = None
        self._tried = False
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return self._load() is not None

    @property
    def loaded(self) -> bool:
        """True once a model is in memory (never triggers a load)."""
        with self._lock:
            return self._model is not None

    def _load(self):
        with self._lock:
            if self._tried:
                return self._model
            # Offline enforcement must patch huggingface_hub.constants too: the
            # flag is snapshotted at import time, so setting the env var after
            # the library is already imported would NOT stop a download.
            with _hf_offline(self.allow_download):
                try:
                    if not self.model_name:
                        return None
                    try:
                        from sentence_transformers import CrossEncoder
                    except ImportError:
                        return None
                    try:
                        self._model = CrossEncoder(self.model_name)
                    except Exception:
                        self._model = None
                finally:
                    # Only mark attempted once the load has finished, so a
                    # concurrent caller blocks on the lock instead of silently
                    # falling back mid-load.
                    self._tried = True
            return self._model

    def score(self, query: str, texts: list[str]) -> list[float] | None:
        """Cross-encoder relevance logits, or None when unavailable."""
        model = self._load()
        if model is None or not texts:
            return None
        try:
            out = model.predict([(query, t) for t in texts])
        except Exception:
            return None
        return [float(x) for x in out]


class NLIEntailmentScorer:
    """Frozen NLI teacher for answerability (CLEAR-style, inference only).

    Scores P(entailment) with premise=chunk, hypothesis=query using a
    DeBERTa-v3 NLI model — same family and task mixture (MNLI/FEVER/ANLI)
    as CLEAR's frozen DeBERTa-v3-large teacher. Loaded lazily and
    offline-first, mirroring :class:`CrossEncoderScorer`: returns None when
    the model is not cached and downloads are disallowed.

    Design caveat: the raw user query (often a question, not a declarative
    hypothesis) is the NLI hypothesis, so P(entail) for "what is X?" is
    semantically approximate — useful as a ranking signal, not a verdict.
    """

    def __init__(self, model_name: str = "", allow_download: bool = False):
        self.model_name = (model_name or "").strip()
        self.allow_download = allow_download
        self._tok = None
        self._model = None
        self._entail_idx = 0
        self._tried = False
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return self._load() is not None

    @property
    def loaded(self) -> bool:
        """True once a model is in memory (never triggers a load)."""
        with self._lock:
            return self._model is not None

    def _load(self):
        with self._lock:
            if self._tried:
                return self._model
            with _hf_offline(self.allow_download):
                try:
                    if not self.model_name:
                        return None
                    try:
                        from transformers import (AutoModelForSequenceClassification,
                                                  AutoTokenizer)
                    except ImportError:
                        return None
                    try:
                        self._tok = AutoTokenizer.from_pretrained(self.model_name)
                        self._model = AutoModelForSequenceClassification.from_pretrained(
                            self.model_name)
                    except Exception:
                        self._tok = None
                        self._model = None
                        return None
                    try:
                        labels = {str(k).lower(): int(v)
                                  for k, v in self._model.config.label2id.items()}
                        self._entail_idx = labels.get("entailment", 0)
                    except Exception:
                        self._entail_idx = 0
                finally:
                    self._tried = True
            return self._model

    def score(self, query: str, texts: list[str]) -> list[float] | None:
        """P(entailment | chunk, query) per chunk, or None when unavailable."""
        model = self._load()
        if model is None or self._tok is None or not texts:
            return None
        try:
            import torch
            import torch.nn.functional as F
        except ImportError:
            return None
        try:
            out: list[float] = []
            with torch.no_grad():
                for i in range(0, len(texts), 8):
                    batch = texts[i:i + 8]
                    enc = self._tok(
                        batch, [query] * len(batch),
                        padding=True, truncation=True, max_length=512,
                        return_tensors="pt",
                    )
                    logits = model(**enc).logits
                    probs = F.softmax(logits, dim=-1)[:, self._entail_idx]
                    out.extend(float(p) for p in probs.tolist())
        except Exception:
            return None
        return out


def sigmoid(xs: list[float]) -> list[float]:
    # Clamp: math.exp overflows for x < -709.
    return [1.0 / (1.0 + math.exp(-max(-500.0, min(500.0, x)))) for x in xs]


def clear_answerability(rel: list[float], entail: list[float],
                        alpha: float = 0.5) -> list[float]:
    """CLEAR inference combination, normalized to [0, 1].

    ``(sigmoid(relevance) + alpha * entailment) / (1 + alpha)``: the paper's
    blend rescaled so the neural and heuristic paths share one scale (the
    raw blend runs to 1 + alpha, which also leaked >100% into the UI).
    Rescaling is monotonic, so rankings are unchanged. Both inputs align
    by position.
    """
    scale = 1.0 + max(0.0, alpha)
    return [(r + alpha * e) / scale for r, e in zip(sigmoid(rel), entail)]


def combine_answerability(cross: list[float] | None,
                          coverage: list[float]) -> list[float]:
    """Blend sigmoid cross-encoder scores with coverage (0.6/0.4);
    coverage alone when no cross-encoder scores are available.

    Sigmoid (absolute) rather than min-max: min-max forces the top
    candidate to 1.0 even when every candidate is irrelevant.
    """
    if cross is None:
        return list(coverage)
    norm = sigmoid(cross)
    return [0.6 * a + 0.4 * b for a, b in zip(norm, coverage)]


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
