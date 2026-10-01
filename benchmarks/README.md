# RAG benchmarks

Offline retrieval benchmarking for procure: **recall@5 / recall@10**,
precision, hit rate, nDCG, MRR, and per-query **latency** (mean/p50/p95).
No downloads, no API keys — the harness runs against the local pipeline
with whatever backend is configured (`hash` embeddings work out of the box).

## Quick start

```sh
# 1. Write the bundled 3-doc / 4-query smoke dataset
procure bench init --init-dir benchmarks/sample

# 2. Run it in an isolated temp library (default; your data untouched)
procure bench run --corpus benchmarks/sample/corpus.jsonl \
  --queries benchmarks/sample/queries.jsonl --markdown

# 3. Save a JSON report for tracking regressions
procure bench run --corpus benchmarks/sample/corpus.jsonl \
  --queries benchmarks/sample/queries.jsonl --out reports/bench.json
```

Useful flags: `--ks 5,10,20` (k values), `--data-dir PATH` (benchmark a
real library instead of a temp one), `--beir-dir DIR` (BEIR-style dataset,
see below), `--markdown` (human-readable summary next to the JSON).

To compare retrieval modes, run twice with different env flags:

```sh
PROCURE_SEARCH=hybrid PROCURE_RERANK=on procure bench run --corpus ... --queries ... --out reports/hybrid.json
PROCURE_SEARCH=dense PROCURE_RERANK=off procure bench run --corpus ... --queries ... --out reports/dense.json
```

## Pulling real BEIR/MTEB datasets

```sh
procure bench list                      # 15 curated sets with sizes
procure bench pull --dataset scifact    # → data/benchmarks/scifact/
procure bench run --corpus data/benchmarks/scifact/corpus.jsonl \
  --queries data/benchmarks/scifact/queries.jsonl \
  --out reports/scifact.json --markdown
```

`pull` downloads from the HuggingFace Hub (`mteb/<repo>` by default —
the BEIR datasets repackaged in classic `corpus.jsonl` / `queries.jsonl` /
`qrels.tsv` layout, readable with stdlib alone) and converts to the
procure format above, plus a `manifest.json` with source URLs and counts.
`--source beir` pulls the `BeIR/*` parquet repos instead (needs
`pip install pyarrow`); `--split test|dev|train` picks the qrels split;
`--max-queries N` keeps the first N judged queries for a quick smoke run
(the corpus always stays complete, so the task never silently shrinks).
Pulled data lands in `data/benchmarks/` (git-ignored), never in the repo.
Start small: `scifact` (~5K docs), `nfcorpus`, `fiqa` — `fever`/`msmarco`
are millions of docs and GBs of download.

## Dataset format

`corpus.jsonl` — one document per line:

```json
{"doc_id": "lighthouse", "filename": "lighthouse.md", "text": "..."}
```

`queries.jsonl` — one query per line, pointing at stable corpus ids:

```json
{"query": "When was the north beacon relamped?", "relevant_doc_ids": ["lighthouse"]}
```

Chunk-level recall is also supported via `relevant_chunk_ids`
(`<doc_id>:<n>` after ingest) — chunk ids win when both are given.
The runner ingests the corpus, rewrites the stable ids to library ids,
then scores `RAGService.search` at each k.

## Open-source benchmarks surveyed

| Benchmark / framework | What it is | What we borrowed |
|---|---|---|
| [BEIR](https://github.com/beir-cellar/beir) | Zero-shot retrieval over 18 datasets (NQ, HotpotQA, FiQA, SciFact…); reports recall@k, MRR, nDCG@k | Metric definitions + the `corpus / queries / qrels` layout (`--beir-dir` reads it directly) |
| [MTEB / MMTEB](https://github.com/embeddings-benchmark/mteb) | Massive embedding benchmark incl. retrieval + reranking tracks | Methodology for comparing embedding backends (`hash` vs `sbert`); run the same dataset under each |
| [RAGAS](https://github.com/explodinggradients/ragas) | Reference-based RAG metrics: context precision/recall, faithfulness, answer relevancy | Metric vocabulary; our context-precision analogue is precision@k over gold docs |
| [DeepEval](https://github.com/confident-ai/deepeval) | CI-friendly LLM evals (contextual precision/recall, faithfulness) | The `--out reports/*.json` pattern: check JSON reports into CI and diff them per release |
| [TruLens](https://github.com/truera/trulens) | Production tracing + groundedness guardrails | Per-query latency + zero-result rate as the production-health half of the report |
| [RAGChecker](https://github.com/amazon-science/RAGChecker) | Fine-grained claim-level RAG diagnosis | Out of scope for the retriever harness; listed for future generation-eval work |
| [ARES](https://github.com/stanford-futuredata/ares) | Lightweight RAG eval with synthetic judgements | The synthetic-query idea: generate queries from your own docs for a cheap private set |
| RGB / CRAG / RAGBench / KILT | End-to-end RAG robustness suites (noise, conflicts, knowledge-intensive NLP) | Future work: they need a generator + judge; the retriever harness here is the prerequisite layer |

Deliberately excluded: anything needing an LLM judge or a download at
eval time (faithfulness, answer relevancy). Those belong in a second,
optional layer on top of the JSON reports produced here.

## Interpreting results

- **recall@k is the ceiling** — if the gold chunk is not in top-k, the
  answer cannot be grounded. Fix recall before reranking.
- **hit rate@k** answers "does at least one gold doc show up"; useful
  for small private sets where recall denominators are 1.
- **nDCG@k / MRR** reward putting gold early — the reranker's job.
  If recall is high but nDCG is low, the retriever finds gold but buries it.
- **p95 latency** matters more than the mean for interactive search;
  compare per-config, on the same machine and backend.
- **zero-result rate** > 0 means queries returning nothing at all —
  usually an over-strict scope filter or an empty library, not a ranking bug.
