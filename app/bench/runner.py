"""Benchmark execution: ingest a corpus, run queries, score + time them.

Two relevance modes per query (chunk-level wins when both are given,
since a chunk id pins the exact gold span while a doc id matches any
chunk of that document):

- ``relevant_chunk_ids`` — retrieved ids are the service's ``chunk_id``.
- ``relevant_doc_ids`` — retrieved ids are the hits' ``doc_id``.
"""
from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field

from app.bench import metrics


@dataclass
class QueryResult:
    query: str
    relevant_doc_ids: list[str] = field(default_factory=list)
    relevant_chunk_ids: list[str] = field(default_factory=list)
    retrieved_ids: list[str] = field(default_factory=list)
    latency_s: float = 0.0
    scores: dict = field(default_factory=dict)


@dataclass
class BenchmarkReport:
    ks: list[int]
    query_results: list[QueryResult]
    summary: dict
    config: dict

    def to_dict(self) -> dict:
        return {
            "config": self.config,
            "ks": self.ks,
            "summary": self.summary,
            "queries": [
                {
                    "query": r.query,
                    "relevant_doc_ids": r.relevant_doc_ids,
                    "relevant_chunk_ids": r.relevant_chunk_ids,
                    "retrieved_ids": r.retrieved_ids,
                    "latency_s": round(r.latency_s, 4),
                    "scores": r.scores,
                }
                for r in self.query_results
            ],
        }

    def to_markdown(self) -> str:
        lines = ["# RAG benchmark report", ""]
        cfg = " · ".join(f"{k}={v}" for k, v in self.config.items())
        lines.append(f"Config: {cfg}")
        lines.append(f"Queries: {self.summary.get('num_queries', 0)}")
        lines.append("")
        lines.append("| k | recall@k | precision@k | hit rate@k | nDCG@k |")
        lines.append("|---|----------|-------------|------------|--------|")
        for k in self.ks:
            m = self.summary.get(f"@{k}", {})
            lines.append(
                f"| {k} | {m.get('recall', 0):.3f} | {m.get('precision', 0):.3f} "
                f"| {m.get('hit_rate', 0):.3f} | {m.get('ndcg', 0):.3f} |")
        lines.append("")
        lat = self.summary.get("latency", {})
        lines.append(
            f"Latency (s): mean {lat.get('mean_s', 0)} · "
            f"p50 {lat.get('p50_s', 0)} · p95 {lat.get('p95_s', 0)} · "
            f"max {lat.get('max_s', 0)} over {lat.get('count', 0)} queries.")
        lines.append(
            f"MRR: {self.summary.get('mrr', 0):.3f} · "
            f"zero-result rate: {self.summary.get('zero_result_rate', 0):.3f}")
        return "\n".join(lines) + "\n"


class BenchmarkRunner:
    """Runs a query set against a live ``RAGService`` and scores it."""

    def __init__(self, service, ks: tuple[int, ...] = (5, 10)):
        self.service = service
        self.ks = sorted({int(k) for k in ks if int(k) > 0} or [5])
        self.depth = max(self.ks)

    def run_query(self, query: str, relevant_doc_ids: list[str],
                  relevant_chunk_ids: list[str]) -> QueryResult:
        """Search once, timing the call; score at every k in ``self.ks``."""
        start = time.perf_counter()
        hits = self.service.search(query, self.depth)
        latency = time.perf_counter() - start
        chunk_mode = bool(relevant_chunk_ids)
        if chunk_mode:
            retrieved = [h["chunk_id"] for h in hits]
            relevant = set(relevant_chunk_ids)
        else:
            retrieved = [h["doc_id"] for h in hits]
            relevant = set(relevant_doc_ids)
        # One doc contributes many chunks, so raw hit lists repeat doc
        # ids — counting every repeat as a gain lets DCG exceed IDCG
        # (nDCG > 1). Score the deduped ranking (first occurrence wins).
        retrieved = list(dict.fromkeys(retrieved))
        scores: dict[str, dict[str, float]] = {}
        for k in self.ks:
            scores[f"@{k}"] = {
                "recall": round(metrics.recall_at_k(retrieved, relevant, k), 4),
                "precision": round(metrics.precision_at_k(retrieved, relevant, k), 4),
                "hit_rate": round(metrics.hit_rate_at_k(retrieved, relevant, k), 4),
                "ndcg": round(metrics.ndcg_at_k(retrieved, relevant, k), 4),
            }
        return QueryResult(
            query=query,
            relevant_doc_ids=list(relevant_doc_ids),
            relevant_chunk_ids=list(relevant_chunk_ids),
            retrieved_ids=retrieved,
            latency_s=latency,
            scores=scores,
        )

    def run(self, queries: list[dict]) -> BenchmarkReport:
        """Score every query and aggregate means + latency + config."""
        results = [self.run_query(q["query"], q.get("relevant_doc_ids") or [],
                                  q.get("relevant_chunk_ids") or [])
                   for q in queries]
        summary: dict = {"num_queries": len(results)}
        for k in self.ks:
            key = f"@{k}"
            summary[key] = {
                "recall": round(statistics.fmean(
                    r.scores[key]["recall"] for r in results), 4),
                "precision": round(statistics.fmean(
                    r.scores[key]["precision"] for r in results), 4),
                "hit_rate": round(statistics.fmean(
                    r.scores[key]["hit_rate"] for r in results), 4),
                "ndcg": round(statistics.fmean(
                    r.scores[key]["ndcg"] for r in results), 4),
            }
        pairs = [(_retrieved_ids(r), _relevant_set(r)) for r in results]
        summary["mrr"] = round(metrics.mrr(pairs), 4)
        summary["latency"] = metrics.latency_stats(
            [r.latency_s for r in results])
        total = len(results)
        summary["zero_result_rate"] = round(
            sum(1 for r in results if not r.retrieved_ids) / total, 4) if total else 0.0
        summary["chunk_level_queries"] = sum(
            1 for r in results if r.relevant_chunk_ids)
        summary["doc_level_queries"] = total - summary["chunk_level_queries"]
        return BenchmarkReport(ks=self.ks, query_results=results,
                               summary=summary, config=_service_config(self.service))


def _retrieved_ids(result: QueryResult) -> list[str]:
    return result.retrieved_ids


def _relevant_set(result: QueryResult) -> set[str]:
    if result.relevant_chunk_ids:
        return set(result.relevant_chunk_ids)
    return set(result.relevant_doc_ids)


def _service_config(service) -> dict:
    settings = getattr(service, "settings", None)
    cfg = {
        "search_mode": getattr(settings, "search_mode", "?"),
        "rerank": getattr(settings, "rerank", "?"),
    }
    try:
        cfg["chunk_count"] = int(service.vectors.count())
    except Exception:
        pass
    try:
        cfg["sparse_chunks"] = int(service.sparse.count())
    except Exception:
        pass
    return cfg


def ingest_corpus(service, corpus: list[dict]) -> dict[str, str]:
    """Ingest corpus docs; returns {corpus doc_id -> library doc_id}.

    Corpus ``doc_id`` values are stable labels (``lighthouse``); library
    doc ids are random hex, so the runner rewrites each query's
    ``relevant_doc_ids`` through this map before scoring.
    """
    mapping = {}
    for doc in corpus:
        res = service.ingest_text(doc["filename"], doc["text"],
                                  doc.get("tags") or [],
                                  doc.get("doc_type") or "document")
        if res.get("status") not in ("ready", "duplicate"):
            raise RuntimeError(
                f"corpus doc {doc['doc_id']!r} failed to ingest: "
                f"{res.get('error', res.get('status'))}")
        mapping[doc["doc_id"]] = res["doc_id"]
    return mapping


def remap_queries(queries: list[dict], mapping: dict[str, str]) -> list[dict]:
    """Rewrite stable corpus doc ids to library doc ids; drops unknowns."""
    out = []
    for q in queries:
        wanted = [mapping[d] for d in q.get("relevant_doc_ids") or [] if d in mapping]
        chunks = list(q.get("relevant_chunk_ids") or [])
        if not wanted and not chunks:
            continue
        out.append({"query": q["query"], "relevant_doc_ids": wanted,
                    "relevant_chunk_ids": chunks})
    return out


def run_benchmark(service, corpus: list[dict], queries: list[dict],
                  ks: tuple[int, ...] = (5, 10)) -> BenchmarkReport:
    """Ingest-then-score convenience entry point used by the CLI and tests."""
    t0 = time.perf_counter()
    mapping = ingest_corpus(service, corpus)
    ingest_s = time.perf_counter() - t0
    runner = BenchmarkRunner(service, ks)
    report = runner.run(remap_queries(queries, mapping))
    report.summary["ingest_s"] = round(ingest_s, 4)
    report.summary["num_docs"] = len(corpus)
    return report
