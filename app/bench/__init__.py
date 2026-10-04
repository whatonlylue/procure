"""Offline RAG retrieval benchmarks: recall@k, MRR, nDCG, latency.

Zero-dependency (stdlib only) harness aligned with the conventions of the
open-source retrieval benchmarks surveyed in ``benchmarks/README.md``
(BEIR for recall@k/MRR/nDCG methodology, MTEB for the embedder angle,
RAGAS/DeepEval/TruLens for the metric vocabulary). LLM-judge metrics
(faithfulness, answer relevancy) are intentionally out of scope: this
harness measures what the retriever controls — did the gold
document/chunk make top-k, and how fast.
"""
from app.bench.datasets import load_corpus_jsonl, load_queries_jsonl
from app.bench.metrics import (
    hit_rate_at_k,
    latency_stats,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from app.bench.runner import BenchmarkRunner, run_benchmark
from app.bench.sources import list_datasets, pull

__all__ = [
    "BenchmarkRunner",
    "hit_rate_at_k",
    "latency_stats",
    "list_datasets",
    "load_corpus_jsonl",
    "load_queries_jsonl",
    "mrr",
    "ndcg_at_k",
    "precision_at_k",
    "pull",
    "recall_at_k",
    "run_benchmark",
]
