"""procure MCP server: search + ingest tools over Streamable HTTP.

Managed from the workspace UI (MCP Server tab), or standalone:

    python -m app.mcp_server [--host 127.0.0.1] [--port 8001]

Clients register the URL printed at startup (http://host:port/mcp).
Reads PROCURE_* env the same way as the main app so both processes share
one data dir, embedding backend, and vector store.
"""
from __future__ import annotations

import argparse
import logging
from typing import Annotated

from mcp.server import MCPServer
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

from app.config import get_settings
from app.service import RAGService
from app.skills import skill_text

logger = logging.getLogger(__name__)

try:
    GUIDE_TEXT = skill_text()
except FileNotFoundError as e:
    logger.warning("procure skill not bundled: %s", e)
    GUIDE_TEXT = ""


def _doc_uri(doc_id: str) -> str:
    return f"procure://documents/{doc_id}"


mcp = MCPServer(
    "procure",
    title="procure RAG",
    description="Local-first retrieval over the procure document library.",
    instructions=(
        "procure stores ingested documents as embedded chunks. Use procure_search "
        "to answer questions from the library, procure_list_documents to see what "
        "is stored, and procure_add_text to persist new material (conversation "
        "transcripts, tool outputs, notes) for future retrieval. Pass "
        'doc_type="memory" to save or search cross-agent memories; omit it to '
        "cover everything. Read the procure://guide resource for the full usage "
        "guide, and fetch any hit's complete document via its `uri` "
        "(procure://documents/{doc_id})."
    ),
)

_READ_ONLY = ToolAnnotations(readOnlyHint=True)


class SearchHit(BaseModel):
    """One retrieved chunk. Typed so clients get a real output schema."""

    chunk_id: str
    doc_id: str
    filename: str
    text: str
    doc_type: str = Field(
        default="document",
        description='"document" or "memory" (cross-agent memory).')
    score: float = Field(description="Blended final ranking score.")
    sparse_score: float = Field(description="BM25 exact-word match strength.")
    dense_score: float = Field(description="Embedding cosine similarity.")
    fused_rank: int = Field(description="Rank after hybrid fusion, before rerank.")
    answerability: float = Field(description="Likelihood this chunk answers the query.")
    entailment: float = Field(description="P(chunk entails query); >=0.5 is a direct answer.")
    uri: str = Field(description="Resource URI for the full document text.")


class LibraryDoc(BaseModel):
    doc_id: str
    filename: str
    status: str
    chunk_count: int
    error: str = ""
    created_at: str = ""
    tags: list[str] = []
    source: str = ""
    doc_type: str = Field(
        default="document",
        description='"document" or "memory" (cross-agent memory).')
    uri: str = Field(description="Resource URI for the full document text.")


class IngestResult(BaseModel):
    doc_id: str
    filename: str
    status: str
    chunk_count: int = 0
    error: str = ""
    doc_type: str = Field(
        default="document",
        description='"document" or "memory" (cross-agent memory).')


_service: RAGService | None = None


def _svc() -> RAGService:
    global _service
    if _service is None:
        _service = RAGService()
        logger.info(
            "RAG service ready: embeddings=%s vectordb=%s chunks=%d documents=%d",
            _service.settings.embeddings, _service.settings.vectordb,
            _service.vectors.count(), len(_service.meta.list()),
        )
    return _service


@mcp.tool(annotations=_READ_ONLY)
def procure_search(
    query: Annotated[str, Field(description="Natural-language question or search query.")],
    top_k: Annotated[int, Field(ge=1, le=50, description="Maximum chunks to return.")] = 5,
    doc_ids: Annotated[
        list[str] | None,
        Field(description="Optional document IDs to scope the search to (see procure_list_documents)."),
    ] = None,
    tags: Annotated[
        list[str] | None,
        Field(description="Optional tags: only chunks from documents carrying ALL of these tags are searched."),
    ] = None,
    doc_type: Annotated[
        str | None,
        Field(description='Optional type filter: "memory" for cross-agent memories only, "document" for files/uploads only, omit for everything.'),
    ] = None,
) -> list[SearchHit]:
    """Advanced search over the procure library: hybrid BM25 + dense retrieval fused with RRF, then reranked by answerability. Returns matching chunks with filenames, relevance scores, and answerability signals. Use this whenever a question could pertain to stored documents, past conversations, or saved outputs. Pass doc_type="memory" when the user references previous work. Each hit's `uri` reads the full document."""
    svc = _svc()
    logger.info("procure_search q=%r top_k=%d doc_ids=%s tags=%s doc_type=%s",
                query[:120], top_k, doc_ids, tags, doc_type)
    return [SearchHit(**h, uri=_doc_uri(h["doc_id"]))
            for h in svc.search(query, top_k, doc_ids, tags, None, doc_type)]


@mcp.tool(annotations=_READ_ONLY)
def procure_list_documents(
    doc_type: Annotated[
        str | None,
        Field(description='Optional type filter: "memory" for cross-agent memories only, "document" for files/uploads only, omit for everything.'),
    ] = None,
) -> list[LibraryDoc]:
    """List every document in the procure library with status and chunk counts. Use it to discover what is stored before searching, or to get document IDs for scoped search. Pass doc_type="memory" to browse only agent memories. Each entry's `uri` reads the full document."""
    docs = [LibraryDoc(**d, uri=_doc_uri(d["doc_id"]))
            for d in _svc().list_documents(doc_type)]
    logger.info("procure_list_documents doc_type=%s -> %d docs", doc_type, len(docs))
    return docs


@mcp.tool()
def procure_add_text(
    title: Annotated[str, Field(description="Short title used as the document name.")],
    text: Annotated[str, Field(description="Full text to store: conversation transcript, tool output, notes, or document content.")],
    tags: Annotated[
        list[str] | None,
        Field(description="Optional tags for later filtered search (e.g. ['project-x'])."),
    ] = None,
    doc_type: Annotated[
        str,
        Field(description='Document type: "memory" to store a cross-agent memory, "document" (default) for ordinary notes and files.'),
    ] = "document",
) -> IngestResult:
    """Add new material to the procure library. The text is chunked, embedded, and stored with the given title, and becomes searchable immediately. Use it to persist conversation transcripts, agent outputs, research notes, or any content worth retrieving later. Pass doc_type="memory" (title "memory: <topic>") for facts worth keeping across agent sessions."""
    logger.info("procure_add_text title=%r chars=%d doc_type=%s",
                title[:80], len(text or ""), doc_type)
    if not (text or "").strip():
        raise ValueError("text must not be empty")
    result = IngestResult(
        **_svc().ingest_text(title.strip() or "snippet", text, tags or [],
                             doc_type))
    logger.info("procure_add_text -> %s", result)
    return result


@mcp.tool()
def procure_add_url(
    url: Annotated[str, Field(description="http(s) URL of an article or page to ingest.")],
    tags: Annotated[
        list[str] | None,
        Field(description="Optional tags for later filtered search."),
    ] = None,
    doc_type: Annotated[
        str,
        Field(description='Document type: "memory" to store a cross-agent memory, "document" (default) for ordinary pages.'),
    ] = "document",
) -> IngestResult:
    """Fetch a web page and add its readable text to the procure library. The article text is extracted, chunked, embedded, and searchable immediately. Use it to persist reference pages worth retrieving later."""
    logger.info("procure_add_url url=%r doc_type=%s", url[:120], doc_type)
    result = IngestResult(**_svc().ingest_url(url, tags or [], doc_type))
    logger.info("procure_add_url -> %s", result)
    return result


@mcp.resource(
    "procure://guide",
    name="procure-guide",
    title="procure agent guide",
    description="How to use the procure library: search, full-document reads, and storing memories, notes, and user files.",
    mime_type="text/markdown",
)
def read_guide() -> str:
    """Usage guide for agents: workflows for search, memories, and storage."""
    logger.info("procure://guide read (%d chars)", len(GUIDE_TEXT))
    return GUIDE_TEXT


@mcp.resource(
    "procure://documents/{doc_id}",
    name="procure-document",
    title="procure document",
    description="Full extracted text of one stored document. Get doc_id values from procure_search hits or procure_list_documents.",
    mime_type="text/plain",
)
def read_document(doc_id: str) -> str:
    """Fetch a complete document after finding it via search or listing."""
    logger.info("procure://documents/%s read", doc_id)
    try:
        doc = _svc().get_document(doc_id)
    except KeyError:
        raise ValueError(f"Unknown document: {doc_id}") from None
    text = doc.get("text", "")
    if not text:
        raise ValueError(f"Document {doc_id} ({doc['filename']}) has no readable text")
    return f"# {doc['filename']}\n\n{text}"


def main(argv: list[str] | None = None) -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="procure MCP server (Streamable HTTP)")
    parser.add_argument("--host", default=settings.mcp_host)
    parser.add_argument("--port", type=int, default=settings.mcp_port)
    args = parser.parse_args(argv)

    url = f"http://{args.host}:{args.port}/mcp"
    # Plain print is fine on HTTP transports (stdout is not the wire here)
    # and gives the workspace UI an immediate first log line.
    print(f"procure MCP server listening on {url}", flush=True)
    logger.info("starting Streamable HTTP server on %s", url)
    mcp.run(transport="streamable-http", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
