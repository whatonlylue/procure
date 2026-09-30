# procure

Local-first personal library: drop in documents, search them by meaning, and let AI agents use the library too. Fully offline by default — no accounts, no cloud.

**Download:** [GitHub Releases](https://github.com/whatonlylue/procure/releases) — macOS (`.dmg`, Apple Silicon) and Windows (installer). Builds are unsigned for now: on macOS, right-click → Open on first launch.

## Features

- **Desktop app** — Tauri shell + Python sidecar that manages its own server; documents live in the OS user-data dir and survive upgrades.
- **Broad ingest** — pdf / docx / pptx / xlsx / epub / odt+ods+odp / rtf / csv / json / xml / md / txt and more, all parsed dependency-free → sentence-aware ~1000-char chunks (150-char overlap, never split mid-sentence) → embedded + stored.
- **Sources** — paste a URL to ingest its article text (trafilatura when installed, readability next, stdlib fallback); watch folders that auto-ingest new files, re-ingest edits in place, and flag deletions as `missing`.
- **Scanned documents** — thin-page detection routes scanned PDFs and images through optional local OCR (RapidOCR preferred, Tesseract fallback): `uv pip install -e '.[ocr]'`.
- **Dedup + background jobs** — sha256 content hashing skips re-uploads; uploads and folder syncs run as cancellable background jobs with progress.
- **Tags + scoped search** — document-level tags inherited by chunks at query time; filter search by tags, source, or an explicit doc set from the Search tab.
- **Hybrid retrieval** — zero-dependency BM25 sparse scores fused with dense cosine via reciprocal-rank fusion (RRF) [1].
- **Answerability reranking** — CLEAR-style inference `sigmoid(relevance) + α · entailment`: a local ms-marco cross-encoder plus a frozen DeBERTa-v3 NLI teacher scoring P(chunk entails query), so answer-bearing chunks outrank topical distractors [2]. Degrades gracefully (cross-encoder + term coverage → coverage alone) when models are unavailable. Retrieval mode and rerank toggle at runtime.
- **Library tab** — per-document status (`ready`/`failed`/`missing`), chunk counts, tags, delete, re-ingest.
- **MCP server tab** — one click starts a Streamable-HTTP Model Context Protocol server exposing `procure_search` (with tag + type scope), `procure_add_text`, `procure_add_url`, `procure_list_documents`, plus `procure://documents/{id}` full-text resources — any agent harness can search, read, and write to the library.
- **Cross-agent memories** — documents typed `memory` are shared across agent sessions and harnesses: agents save with `procure_add_text(..., doc_type="memory")` and recall with `procure_search(..., doc_type="memory")` whenever the user references previous work.
- **Pluggable backends** — embeddings: `hash` (zero-dep) / `sbert` (local neural, recommended) / `openai`; vector stores: `sqlite` / `chroma`. All via env flags.

On a 10-book / 10-query eval, hybrid + CLEAR rerank reaches **9/10 top-1 (MRR 0.950)** with `sbert`, vs 6/10 dense-only.

## Run from source

```sh
uv pip install -e '.[sbert]'   # or -e . for zero-dep defaults
cd frontend && npm install && npm run build && cd ..
PROCURE_EMBEDDINGS=sbert uv run uvicorn app.main:app --port 8000
# open http://127.0.0.1:8000
```

Rebuild the desktop app after changing code: `./scripts/build_desktop.sh` (see [PACKAGING.md](PACKAGING.md)).

## MCP for agents

Start the server from the MCP tab (or `uv run python -m app.mcp_server --port 8001`), then register it:

```sh
Muse mcp add --transport http procure http://127.0.0.1:8001/mcp
```

Then install the agent skill so harnesses proactively search the library and save memories — click **Install skill** in the MCP tab, or from source:

```sh
uv run python -m app.cli install-skills
```

Claude Code users can instead install the bundled plugin (`/plugin marketplace add whatonlylue/procure`, then `/plugin install procure`), which wires the MCP server and the skill together. The same content is always served live as the `procure://guide` resource.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `PROCURE_EMBEDDINGS` | `hash` | `hash` · `sbert` (recommended) · `openai` |
| `PROCURE_VECTORDB` | `sqlite` | `sqlite` · `chroma` |
| `PROCURE_SEARCH` / `PROCURE_RERANK` | `hybrid` / `on` | retrieval mode · answerability rerank |
| `PROCURE_TOPK` | `5` | default hits per query |
| `PROCURE_CROSS_ENCODER_ALLOW_DOWNLOAD` / `PROCURE_NLI_ALLOW_DOWNLOAD` | `0` | set `1` once to fetch neural models, then stays offline |
| `PROCURE_DATA_DIR` | `./data` | raw files + databases (desktop app uses the OS user-data dir) |
| `PROCURE_OCR` | `auto` | `auto` · `on` · `off` — OCR for scanned PDFs/images (needs `[ocr]` extra) |
| `PROCURE_WATCH_INTERVAL` | `60` | folder-watch poll seconds (`0` disables the loop; manual sync still works) |
| `PROCURE_FETCH_TIMEOUT` | `20` | URL fetch timeout in seconds |

## References

[1] Cormack, Clarke & Buettcher — *Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods* (SIGIR 2009).
[2] Qin et al. — *From Topical Relevance to Answerability: Entailment Distillation for Conversational Retrieval* (CLEAR, arXiv:2609.03482) — [paper](https://arxiv.org/abs/2609.03482) · [code](https://github.com/HAI-UESTC/CLEAR). We reimplement only the published inference formula (`sigmoid(relevance) + α·entailment`, dual-head relevance + entailment structure, DeBERTa-v3 NLI teacher family); their trained entailment head and abductive-recall channel are not included. No CLEAR code or weights are copied.

## API (dev)

`GET /api/health` · `POST /api/documents/upload[?async=1]` · `POST /api/documents/url` · `GET /api/documents` (+ `/{id}`) · `PATCH /api/documents/{id}/tags` · `DELETE /api/documents/{id}` (+ `/reingest`) · `GET /api/tags` · `GET /api/search?q=…[&tag=…&source=…&doc_id=…&doc_type=…]` · `/api/jobs[/{id}]` · `/api/watch[/{id}|/sync]` · `PATCH /api/settings` · `/api/mcp/{status,start,stop,logs}` · `/api/skills/{status,install}`

## License

MIT — see [LICENSE](LICENSE).
