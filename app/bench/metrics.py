"""Standard IR retrieval metrics (BEIR methodology, binary relevance).

All rank metrics take an ordered list of retrieved ids and a set of
relevant ids. Ranks are 1-based. Empty relevant sets score 0.0 (a query
with no gold answer cannot be recalled); callers that want to skip such
queries should filter them before aggregating.
"""
from __future__ import annotations

import math
import statistics


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant ids present in the top-k retrieved."""
    if not relevant or k <= 0:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of the top-k retrieved that is relevant."""
    if k <= 0:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / k


def hit_rate_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """1.0 when at least one relevant id is in the top-k, else 0.0."""
    if not relevant or k <= 0:
        return 0.0
    return 1.0 if set(retrieved[:k]) & relevant else 0.0


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    """1/rank of the first relevant hit (0.0 when there is none)."""
    for rank, cid in enumerate(retrieved, start=1):
        if cid in relevant:
            return 1.0 / rank
    return 0.0


def mrr(results: list[tuple[list[str], set[str]]]) -> float:
    """Mean reciprocal rank over (retrieved, relevant) pairs."""
    if not results:
        return 0.0
    return sum(reciprocal_rank(r, rel) for r, rel in results) / len(results)


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """nDCG@k with binary relevance (1 for gold, 0 otherwise)."""
    if not relevant or k <= 0:
        return 0.0
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, cid in enumerate(retrieved[:k], start=1)
        if cid in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def latency_stats(latencies: list[float]) -> dict:
    """Mean/p50/p95/min/max over per-query seconds (plus count)."""
    if not latencies:
        return {"count": 0, "mean_s": 0.0, "p50_s": 0.0, "p95_s": 0.0,
                "min_s": 0.0, "max_s": 0.0}
    ordered = sorted(latencies)
    # Nearest-rank percentiles on the sorted samples (no interpolation),
    # so p50/p95 are always values actually observed.
    p50 = ordered[min(len(ordered) - 1, int(math.ceil(0.50 * len(ordered))) - 1)]
    p95 = ordered[min(len(ordered) - 1, int(math.ceil(0.95 * len(ordered))) - 1)]
    return {
        "count": len(latencies),
        "mean_s": round(statistics.fmean(latencies), 4),
        "p50_s": round(p50, 4),
        "p95_s": round(p95, 4),
        "min_s": round(ordered[0], 4),
        "max_s": round(ordered[-1], 4),
    }
