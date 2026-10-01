---
name: procure
description: Search and manage the procure personal library (documents, notes, web pages) through MCP tools, and save or recall cross-agent memories. Use when answering from stored material, when the user references previous work or past conversations, or when durable facts worth keeping across sessions arise.
license: MIT
compatibility: Requires the procure app with its MCP server running (default http://127.0.0.1:8001/mcp).
metadata:
  version: "1.0.0"
---

# procure

procure is a local-first personal library: ingested documents are split into
chunks, embedded, and stored for hybrid (keyword + meaning) search with an
answerability reranker. This guide is also served live as the
`procure://guide` MCP resource.

Documents have a type: `document` (files, uploads, pages, ordinary notes)
or `memory` (cross-agent memories readable by future sessions in any
harness). Filter by type whenever you only want one kind.

## Tools

- `procure_search(query, top_k=5, doc_ids=None, tags=None,
  doc_type=None, source=None, since=None)` — hybrid BM25 + dense
  retrieval fused with RRF, reranked by answerability. Every hit carries
  `doc_id`, `filename`, `doc_type`, `chunk_id`, `text`, `score`,
  `sparse_score`, `dense_score`, `fused_rank`, `answerability`,
  `entailment`, and a `uri` for the full document. `entailment >= 0.5`
  means the chunk directly supports an answer (neural NLI only; it is
  0 when the NLI model isn't loaded, e.g. the desktop default — use
  `answerability` then). Use `doc_ids` to scope the search after
  listing, `tags` to search only documents carrying all of those tags,
  `source` (upload/text/url/watch) to scope by origin, `since` (ISO
  date, e.g. 2026-01-15) for recent documents, or `doc_type="memory"`
  to search only agent memories (`"document"` for everything else;
  omit for all types).
- `procure_list_documents(doc_type=None, source=None, query=None,
  limit=50, offset=0)` — documents with `doc_id`, `filename`, `status`,
  `chunk_count`, `tags`, `source`, `doc_type`, and `uri`. Call it first
  when you need to discover what exists or to resolve a `doc_id`. Pass
  `doc_type="memory"` to browse only agent memories. Page large
  libraries with `limit`/`offset`; filter by filename with `query`.
- `procure_add_text(title, text, tags=None, doc_type="document")` —
  persist new material (conversation transcripts, tool outputs, research
  notes, user files pasted as text). It is chunked, embedded, and
  searchable immediately. Titles become the document name (`.md` is
  appended when no text extension is present). Pass
  `doc_type="memory"` with a title like `memory: <topic>` to store a
  cross-agent memory; tag it by subject (e.g. `project-x`) so later
  searches can filter by tag as well as type.
- `procure_add_url(url, tags=None, doc_type="document")` — fetch one
  web page and persist its readable article text as a document. Same
  pipeline as pasted text.
- `procure_update_text(doc_id, text, title=None, tags=None)` — replace
  a stored text in place (re-chunked, re-embedded). Use it to correct
  or supersede a memory or note instead of letting stale material
  accumulate.
- `procure_delete_document(doc_id)` — delete a document and its chunks.
  Use it to remove stale, wrong, or superseded memories. Cannot be
  undone.

## Resources

- `procure://documents/{doc_id}` — the full extracted text of one document.
  Search returns only matching chunks; read this resource when the user
  asks for the whole document, when chunks look truncated, or when you
  need complete context before quoting or summarizing.
- `procure://guide` — this guide.

## Workflows

**Answering from the library.** Search first (`top_k` 5–10). If no hit has
`entailment >= 0.5`, try a rephrased query or list documents and scope the
search. Read `procure://documents/{doc_id}` for full context when a chunk
is promising but incomplete. Cite the `filename` in answers.

**Saving memories (when this MCP server is available).** When the user
shares something worth keeping across sessions — preferences, decisions,
project context, environment facts, useful outputs, todos — store it with
`procure_add_text` and `doc_type="memory"` under a short descriptive
title such as `memory: <topic>`. Keep one topic per document so future
searches stay precise. Memories are shared across agents and harnesses:
write them so another session can act on them without extra context.
Confirm briefly what was stored.

**Recalling memories (when this MCP server is available).** When the user
references previous work — "last time", "we discussed", "remember",
"my project", "earlier", "before", "our decision", or any question prior
context could answer — search memories first with
`procure_search(query, doc_type="memory")` before answering from the
general library or from scratch. If no memory hit has `entailment >=
0.5` (or answerability is low), broaden to the full library (omit
`doc_type`), try a rephrased query, or list memories with
`procure_list_documents(doc_type="memory")`.

**Revising memories.** When a stored memory is wrong or outdated, fix it
with `procure_update_text` (or remove it with
`procure_delete_document`) rather than adding a corrected copy — search
would otherwise keep surfacing the stale version.

**Storing user files.** If the user pastes file content, store it with
`procure_add_text` using the original filename as the title. Large
pastes are fine; they are chunked automatically. Binary files (pdf, docx,
pptx) cannot arrive through this server — ask the user to drop them in
the procure workspace UI instead, where they are text-extracted.

## Limits

- File uploads (pdf, docx, pptx) cannot arrive through this server —
  ask the user to drop them in the procure workspace UI instead, where
  they are text-extracted.
- Search matches stored text only; `procure_add_url` fetches single
  pages on demand but this server is not a general web browser, and it
  cannot read files outside the library. Private/intranet URLs are
  refused unless the server allows them.
- Retrieval settings (matching mode, answer ranking) are configured on
  the server; every search uses the same pipeline as the workspace UI.
