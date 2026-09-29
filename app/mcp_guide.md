# procure agent guide

procure is a local-first personal library: ingested documents are split into
chunks, embedded, and stored for hybrid (keyword + meaning) search with an
answerability reranker. This guide is also served live as the
`procure://guide` resource.

## Tools

- `procure_search(query, top_k=5, doc_ids=None)` — hybrid BM25 + dense
  retrieval fused with RRF, reranked by answerability. Every hit carries
  `doc_id`, `filename`, `chunk_id`, `text`, `score`, `sparse_score`,
  `dense_score`, `fused_rank`, `answerability`, `entailment`, and a `uri`
  for the full document. `entailment >= 0.5` means the chunk directly
  supports an answer. Use `doc_ids` to scope a search after listing.
- `procure_list_documents()` — every stored document with `doc_id`,
  `filename`, `status`, `chunk_count`, and `uri`. Call it first when you
  need to discover what exists or to resolve a `doc_id`.
- `procure_add_text(title, text)` — persist new material (conversation
  transcripts, tool outputs, research notes, user files pasted as text).
  It is chunked, embedded, and searchable immediately. Titles become the
  document name (`.md` is appended when no text extension is present).

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

**Making memories.** When the user shares something worth keeping
(preferences, decisions, project notes, useful outputs), store it with
`procure_add_text` under a short descriptive title such as
`memory: <topic>` or `notes: <topic>`. Keep one topic per document so
future searches stay precise. Confirm briefly what was stored.

**Storing user files.** If the user pastes file content, store it with
`procure_add_text` using the original filename as the title. Large
pastes are fine; they are chunked automatically. Binary files (pdf, docx,
pptx) cannot arrive through this server — ask the user to drop them in
the procure workspace UI instead, where they are text-extracted.

## Limits

- The library owner curates documents in the workspace UI; this server
  cannot delete or re-ingest them.
- Search matches stored text only; it cannot browse the web or read files
  outside the library.
- Retrieval settings (matching mode, answer ranking) are configured on
  the server; every search uses the same pipeline as the workspace UI.
