# procure

> Local-first personal library: drop in documents, search them by meaning, and let AI agents use the library too. Fully offline by default — no accounts, no cloud. Provides cross agent memory support to share memories amongst your agents

**Download:** [GitHub Releases](https://github.com/whatonlylue/procure/releases) — macOS (`.dmg`, Apple Silicon) and Windows (installer). Builds are unsigned for now: on macOS, right-click → Open on first launch.

## Features

- **Desktop app** — Tauri shell + Python sidecar that manages its own server; documents live in the OS user-data dir and survive upgrades.
- **Broad ingest** — pdf / docx / pptx / xlsx / epub / odt+ods+odp / rtf / csv / json / xml / md / txt and more, all parsed dependency-free → sentence-aware ~1000-char chunks (150-char overlap, never split mid-sentence) → embedded + stored.
- **Sources** — drag-and-drop files (or browse) to ingest them into the library.
- **Scanned documents** — thin-page detection routes scanned PDFs and images through optional local OCR (RapidOCR preferred, Tesseract fallback): `uv pip install -e '.[ocr]'`.
- **Dedup + background jobs** — sha256 content hashing skips re-uploads; uploads run as cancellable background jobs with progress.
- **Tags + scoped search** — document-level tags inherited by chunks at query time; filter search by tags, type, or an explicit doc set from the Search tab.
- **Hybrid retrieval** — zero-dependency BM25 sparse scores fused with dense cosine via reciprocal-rank fusion (RRF) [1].
- **Answerability reranking** — CLEAR-style inference `sigmoid(relevance) + α · entailment`: a local ms-marco cross-encoder plus a frozen DeBERTa-v3 NLI teacher scoring P(chunk entails query), so answer-bearing chunks outrank topical distractors [2]. Degrades gracefully (cross-encoder + term coverage → coverage alone) when models are unavailable. Retrieval mode and rerank toggle at runtime.
- **Library tab** — per-document status (`ready`/`failed`), chunk counts, tags, delete, re-ingest.
- **MCP server tab** — one click starts a Streamable-HTTP Model Context Protocol server exposing `procure_search` (tag + type + recency scope), `procure_add_text`, `procure_update_text`, `procure_delete_document`, `procure_list_documents` (paged + filtered), plus `procure://documents/{id}` full-text resources — any agent harness can search, read, write, revise, and delete in the library.
- **Cross-agent memories** — documents typed `memory` are shared across agent sessions and harnesses: agents save with `procure_add_text(..., doc_type="memory")` and recall with `procure_search(..., doc_type="memory")` whenever the user references previous work.
- **Local-only pipeline** — granite-embedding-small-english-r2 (384-dim, Apache-2.0, fp16 ONNX) served by ONNX Runtime on whatever GPU is present (CUDA · CoreML · DirectML · ROCm · OpenVINO, CPU fallback) plus a SQLite vector store; no accounts, no cloud. The ~100MB model downloads once into the HF cache on first ingest.

On a 10-book / 10-query eval, hybrid + CLEAR rerank reaches **9/10 top-1 (MRR 0.950)**, vs 6/10 dense-only.

## Run from source

```sh
uv pip install -e .
cd frontend && npm install && npm run build && cd ..
uv run uvicorn app.main:app --port 8000
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

The same content is always served live as the `procure://guide` resource.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `PROCURE_EMBEDDING_PROVIDERS` | auto | comma-separated ONNX provider override, e.g. `cuda` or `cpu` (auto picks the best available: CUDA → CoreML → ROCm/MIGraphX → DirectML → OpenVINO → CPU; NVIDIA GPUs need `onnxruntime-gpu` instead of `onnxruntime`) |
| `PROCURE_EMBEDDING_FILE` | `onnx/model_fp16.onnx` | model file inside the repo (`onnx/model.onnx` is fp32; `onnx/model_quantized.onnx` is smaller/CPU-lean) |
| `PROCURE_EMBEDDING_DIR` | HF cache | pre-downloaded model repo checkout (offline use) |
| `PROCURE_EMBEDDING_BATCH` / `PROCURE_EMBEDDING_MAX_LENGTH` | `32` / `2048` | texts per inference step · token cap per text (model max 8192) |
| `PROCURE_SEARCH` / `PROCURE_RERANK` | `hybrid` / `on` | retrieval mode · answerability rerank |
| `PROCURE_TOPK` | `5` | default hits per query |
| `PROCURE_CROSS_ENCODER_ALLOW_DOWNLOAD` / `PROCURE_NLI_ALLOW_DOWNLOAD` | `0` | set `1` once to fetch neural models, then stays offline |
| `PROCURE_DATA_DIR` | OS user-data dir | raw files + databases (same default for the app, CLI, and MCP server) |
| `PROCURE_OCR` | `auto` | `auto` (scanned pages) · `on` (every PDF page) · `off` — OCR for scanned PDFs/images (needs `[ocr]` extra) |
| `PROCURE_MAX_UPLOAD_MB` | `100` | per-file upload cap in megabytes (413 beyond it) |

## References

[1] Cormack, Clarke & Buettcher — *Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods* (SIGIR 2009).
[2] Qin et al. — *From Topical Relevance to Answerability: Entailment Distillation for Conversational Retrieval* (CLEAR, arXiv:2609.03482) — [paper](https://arxiv.org/abs/2609.03482) · [code](https://github.com/HAI-UESTC/CLEAR). We reimplement only the published inference formula (`sigmoid(relevance) + α·entailment`, dual-head relevance + entailment structure, DeBERTa-v3 NLI teacher family); their trained entailment head and abductive-recall channel are not included. No CLEAR code or weights are copied.

## API (dev)

`GET /api/health` · `POST /api/documents/upload[?async=1]` · `GET /api/documents[?q=&doc_type=&sort=&limit=&offset=]` (+ `/{id}`, `PATCH /{id}` for filename/doc_type) · `PATCH /api/documents/{id}/tags` · `DELETE /api/documents/{id}` (+ `/reingest[?async=1]`) · export/import (`GET /api/documents/export`, `POST /api/documents/import`) · `GET /api/tags` · `GET /api/search?q=…[&tag=…&doc_id=…&doc_type=…&since=…]` · `/api/jobs[/{id}]` · `PATCH /api/settings` · `/api/mcp/{status,start,stop,logs,tools}` · `/api/logs` · `/api/skills/{status,install}`

## CLI (dev)

`procure --version` · `procure serve [--port 8000] [--data-dir …]` · `procure mcp-server` · `procure add <file|-> [--tags …] [--doc-type …]` · `procure search <query> [--top-k …]` · `procure list [--query …]` · `procure install-skills` / `skills-status` / `uninstall-skills`

## License

MIT — see [LICENSE](LICENSE).
