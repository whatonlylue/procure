"""Hybrid search primitives: zero-dependency BM25 sparse store, RRF fusion,
and a CLEAR-style answerability reranker (relevance + NLI entailment).

CLEAR (arXiv:2609.03482, "From Topical Relevance to Answerability") ranks
chunks at inference as ``sigmoid(relevance) + alpha * entailment``: a
relevance discriminator plus an NLI teacher's P(entailment | chunk, query),
so answer-supporting chunks outrank topical distractors. Their trained
entailment head needs TopiOCQA/QReCC supervision we don't have, so we use
the teacher family directly: a public DeBERTa-v3 NLI model (same family and
task mixture as the paper's frozen DeBERTa-v3-large teacher), with the
paper's ms-marco cross-encoder as the relevance head. The abductive-recall
channel (per-passage LLM query generation) is intentionally not faked: the
BM25+dense dual channel already covers candidate-pool breadth.

All sparse scoring here is dependency-free (stdlib + sqlite3). The neural
paths load lazily and offline-first; when a model is unavailable scoring
falls back to the coverage heuristic.
"""
from __future__ import annotations

import math
import os
import re
import sqlite3
from dataclasses import dataclass

_WORD = re.compile(r"[a-z0-9]+")
RRF_K = 60

# Minimal stopword list so coverage scoring is driven by content terms.
_STOPWORDS = frozenset(
    "a an the and or but if then else when what what which who whom whose why how "
    "is are was were be been being do does did done have has had having will would "
    "can could shall should may might must of in on at to for with by from as into "
    "it its this that these those there their them they he she we you i me my our "
    "your his her our us about over under again once here".split()
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


class SparseStore:
    """Incremental BM25 stats over stored chunks (sqlite, zero extra deps).

    Chunk term frequencies live in ``postings``; document frequency is derived
    per query term, so ingest/delete/reingest stay incremental with no global
    rebuild.
    """

    K1 = 1.5
    B = 0.75

    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        with self._connect() as c:
            c.execute(
                """CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    length INTEGER NOT NULL DEFAULT 0
                )"""
            )
            c.execute(
                """CREATE TABLE IF NOT EXISTS postings (
                    chunk_id TEXT NOT NULL,
                    term TEXT NOT NULL,
                    tf INTEGER NOT NULL,
                    PRIMARY KEY (chunk_id, term)
                )"""
            )
            c.execute("CREATE INDEX IF NOT EXISTS idx_post_term ON postings(term)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id)")

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def upsert_chunks(self, ids: list[str], doc_ids: list[str], texts: list[str]) -> None:
        with self._connect() as c:
            for cid, did, text in zip(ids, doc_ids, texts):
                toks = tokenize(text)
                c.execute("DELETE FROM postings WHERE chunk_id=?", (cid,))
                c.execute(
                    """INSERT INTO chunks (chunk_id, doc_id, text, length)
                       VALUES (?, ?, ?, ?)
                       ON CONFLICT(chunk_id) DO UPDATE SET
                         doc_id=excluded.doc_id, text=excluded.text,
                         length=excluded.length""",
                    (cid, did, text, len(toks)),
                )
                tf: dict[str, int] = {}
                for t in toks:
                    tf[t] = tf.get(t, 0) + 1
                c.executemany(
                    "INSERT INTO postings (chunk_id, term, tf) VALUES (?, ?, ?)",
                    [(cid, t, n) for t, n in tf.items()],
                )

    def delete_by_doc(self, doc_id: str) -> int:
        with self._connect() as c:
            ids = [r[0] for r in c.execute(
                "SELECT chunk_id FROM chunks WHERE doc_id=?", (doc_id,)).fetchall()]
            if ids:
                ph = ",".join("?" for _ in ids)
                c.execute(f"DELETE FROM postings WHERE chunk_id IN ({ph})", ids)
                c.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
        return len(ids)

    def count(self) -> int:
        with self._connect() as c:
            return int(c.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

    def _corpus_stats(self, c: sqlite3.Connection) -> tuple[int, float]:
        row = c.execute("SELECT COUNT(*), AVG(length) FROM chunks").fetchone()
        return int(row[0] or 0), float(row[1] or 0.0)

    def idf_map(self, terms: list[str]) -> dict[str, float]:
        """IDF per term under current stats: ln(1 + (N - df + 0.5) / (df + 0.5))."""
        terms = list(dict.fromkeys(terms))
        if not terms:
            return {}
        with self._connect() as c:
            n, _ = self._corpus_stats(c)
            if n == 0:
                return {}
            ph = ",".join("?" for _ in terms)
            df = {t: 0 for t in terms}
            for term, cnt in c.execute(
                f"SELECT term, COUNT(DISTINCT chunk_id) FROM postings "
                f"WHERE term IN ({ph}) GROUP BY term", terms,
            ).fetchall():
                df[term] = int(cnt)
        return {t: math.log(1.0 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def search(self, query: str, top_n: int,
               doc_ids: list[str] | None = None) -> list[SparseHit]:
        terms = content_terms(query)
        if not terms:
            return []
        with self._connect() as c:
            n, avgdl = self._corpus_stats(c)
            if n == 0 or avgdl <= 0:
                return []
            ph = ",".join("?" for _ in terms)
            df = {t: 0 for t in terms}
            for term, cnt in c.execute(
                f"SELECT term, COUNT(DISTINCT chunk_id) FROM postings "
                f"WHERE term IN ({ph}) GROUP BY term", terms,
            ).fetchall():
                df[term] = int(cnt)
            idf = {t: math.log(1.0 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
            if doc_ids:
                dph = ",".join("?" for _ in doc_ids)
                rows = c.execute(
                    f"SELECT p.chunk_id, p.term, p.tf, ch.doc_id, ch.text, ch.length "
                    f"FROM postings p JOIN chunks ch ON ch.chunk_id = p.chunk_id "
                    f"WHERE p.term IN ({ph}) AND ch.doc_id IN ({dph})",
                    terms + list(doc_ids),
                ).fetchall()
            else:
                rows = c.execute(
                    f"SELECT p.chunk_id, p.term, p.tf, ch.doc_id, ch.text, ch.length "
                    f"FROM postings p JOIN chunks ch ON ch.chunk_id = p.chunk_id "
                    f"WHERE p.term IN ({ph})",
                    terms,
                ).fetchall()
        scores: dict[str, float] = {}
        meta: dict[str, tuple[str, str]] = {}
        for cid, term, tf, did, text, length in rows:
            norm = self.K1 * (1.0 - self.B + self.B * (length / avgdl))
            scores[cid] = scores.get(cid, 0.0) + idf[term] * (tf * (self.K1 + 1.0)) / (tf + norm)
            meta[cid] = (did, text)
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
        return [SparseHit(cid, meta[cid][0], meta[cid][1], s) for cid, s in ranked]


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

    @property
    def available(self) -> bool:
        return self._load() is not None

    def _load(self):
        if self._tried:
            return self._model
        self._tried = True
        if not self.model_name:
            return None
        # Offline enforcement must patch huggingface_hub.constants too: the
        # flag is snapshotted at import time, so setting the env var after
        # the library is already imported would NOT stop a download.
        restore_env = os.environ.get("HF_HUB_OFFLINE")
        hf_const = None
        restore_const = None
        if not self.allow_download:
            os.environ["HF_HUB_OFFLINE"] = "1"
            try:
                import huggingface_hub.constants as hf_const
                restore_const = hf_const.HF_HUB_OFFLINE
                hf_const.HF_HUB_OFFLINE = True
            except ImportError:
                hf_const = None
        try:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError:
                return None
            try:
                self._model = CrossEncoder(self.model_name)
            except Exception:
                self._model = None
        finally:
            if not self.allow_download:
                if restore_env is None:
                    os.environ.pop("HF_HUB_OFFLINE", None)
                else:
                    os.environ["HF_HUB_OFFLINE"] = restore_env
                if hf_const is not None:
                    hf_const.HF_HUB_OFFLINE = restore_const
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
    as CLEAR's frozen DeBERTa-v3-large teacher (their Appendix F shows a
    base-size teacher transfers most of the signal). Loaded lazily and
    offline-first, mirroring :class:`CrossEncoderScorer`: returns None when
    the model is not cached and downloads are disallowed.
    """

    def __init__(self, model_name: str = "", allow_download: bool = False):
        self.model_name = (model_name or "").strip()
        self.allow_download = allow_download
        self._tok = None
        self._model = None
        self._entail_idx = 0
        self._tried = False

    @property
    def available(self) -> bool:
        return self._load() is not None

    def _load(self):
        if self._tried:
            return self._model
        self._tried = True
        if not self.model_name:
            return None
        restore_env = os.environ.get("HF_HUB_OFFLINE")
        hf_const = None
        restore_const = None
        if not self.allow_download:
            os.environ["HF_HUB_OFFLINE"] = "1"
            try:
                import huggingface_hub.constants as hf_const
                restore_const = hf_const.HF_HUB_OFFLINE
                hf_const.HF_HUB_OFFLINE = True
            except ImportError:
                hf_const = None
        try:
            try:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
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
            if not self.allow_download:
                if restore_env is None:
                    os.environ.pop("HF_HUB_OFFLINE", None)
                else:
                    os.environ["HF_HUB_OFFLINE"] = restore_env
                if hf_const is not None:
                    hf_const.HF_HUB_OFFLINE = restore_const
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
    return [1.0 / (1.0 + math.exp(-x)) for x in xs]


def clear_answerability(rel: list[float], entail: list[float],
                        alpha: float = 0.5) -> list[float]:
    """CLEAR inference combination (mode "all"): sigmoid(relevance logits)
    plus alpha times entailment probability. Both inputs align by position."""
    return [r + alpha * e for r, e in zip(sigmoid(rel), entail)]


def combine_answerability(cross: list[float] | None,
                          coverage: list[float]) -> list[float]:
    """Blend min-max normalized cross-encoder scores with coverage (0.6/0.4);
    coverage alone when no cross-encoder scores are available."""
    if cross is None:
        return list(coverage)
    lo, hi = min(cross), max(cross)
    if hi > lo:
        norm = [(x - lo) / (hi - lo) for x in cross]
    else:
        norm = [1.0 for _ in cross]
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
    if len(q) > 1:
        qb = set(zip(q, q[1:]))
        cb = set(zip(c, c[1:]))
        bonus = 0.25 * len(qb & cb) / len(qb)
    return min(1.0, base + bonus)
