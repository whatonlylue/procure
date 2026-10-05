"""procure MCP server: search + ingest tools over Streamable HTTP.

Managed from the workspace UI (MCP Server tab), or standalone:

    python -m app.mcp_server [--host 127.0.0.1] [--port 8001]

Clients register the URL printed at startup (http://host:port/mcp).
Reads PROCURE_* env the same way as the main app so both processes share
one data dir and library.
"""
from __future__ import annotations

import argparse
import logging
import threading
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
        "is stored, procure_add_text to persist new material (conversation "
        "transcripts, tool outputs, notes) for future retrieval, "
        "procure_update_text to revise a stored text, and "
        "procure_delete_document to remove one. Pass "
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
    uri: str = Field(description="Resource URI for the full document text.")


class LibraryDoc(BaseModel):
    doc_id: str
    filename: str
    status: str
    chunk_count: int
    error: str = ""
    created_at: str = ""
    tags: list[str] = []
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
_svc_lock = threading.Lock()


def _svc() -> RAGService:
    global _service
    if _service is None:
        # Sync tools run in a threadpool: guard the lazy singleton so two
        # concurrent calls can't build two RAGService instances.
        with _svc_lock:
            if _service is None:
                _service = RAGService()
                logger.info(
                    "RAG service ready: chunks=%d documents=%d",
                    _service.vectors.count(), _service.meta.count(),
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
    since: Annotated[
        str | None,
        Field(description="Optional recency filter: only documents created at/after this ISO date (e.g. 2026-01-15)."),
    ] = None,
) -> list[SearchHit]:
    """Advanced search over the procure library: hybrid BM25 + dense retrieval fused with RRF, then reranked with a cross-encoder. Returns matching chunks with filenames and relevance scores. Use this whenever a question could pertain to stored documents, past conversations, or saved outputs. Pass doc_type="memory" when the user references previous work. Each hit's `uri` reads the full document."""
    svc = _svc()
    logger.info("procure_search q=%r top_k=%d doc_ids=%s tags=%s doc_type=%s since=%s",
                query[:120], top_k, doc_ids, tags, doc_type, since)
    return [SearchHit(**h, uri=_doc_uri(h["doc_id"]))
            for h in svc.search(query, top_k, doc_ids, tags,
                                doc_type, since)]


@mcp.tool(annotations=_READ_ONLY)
def procure_list_documents(
    doc_type: Annotated[
        str | None,
        Field(description='Optional type filter: "memory" for cross-agent memories only, "document" for files/uploads only, omit for everything.'),
    ] = None,
    query: Annotated[
        str | None,
        Field(description="Optional filename substring filter."),
    ] = None,
    limit: Annotated[
        int,
        Field(ge=1, le=500, description="Maximum documents to return."),
    ] = 50,
    offset: Annotated[
        int,
        Field(ge=0, description="Documents to skip (paging)."),
    ] = 0,
) -> list[LibraryDoc]:
    """List documents in the procure library with status and chunk counts. Use it to discover what is stored before searching, or to get document IDs for scoped search. Pass doc_type="memory" to browse only agent memories. Page large libraries with limit/offset. Each entry's `uri` reads the full document."""
    docs = [LibraryDoc(**d, uri=_doc_uri(d["doc_id"]))
            for d in _svc().list_documents(doc_type, query,
                                           limit, offset)]
    logger.info("procure_list_documents doc_type=%s query=%s limit=%d offset=%d -> %d docs",
                doc_type, query, limit, offset, len(docs))
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
def procure_update_text(
    doc_id: Annotated[str, Field(description="Document ID to update (see procure_list_documents).")],
    text: Annotated[str, Field(description="Replacement full text: the document is re-chunked and re-embedded.")],
    title: Annotated[
        str | None,
        Field(description="Optional new title (renames the document)."),
    ] = None,
    tags: Annotated[
        list[str] | None,
        Field(description="Optional new tags (replaces existing tags when given)."),
    ] = None,
) -> IngestResult:
    """Replace a stored text's content in place. Use it to correct or supersede a memory or note: stale material is revised instead of accumulating forever. Only texts ingested via procure_add_text should be updated this way."""
    logger.info("procure_update_text doc_id=%r chars=%d", doc_id, len(text or ""))
    if not (text or "").strip():
        raise ValueError("text must not be empty")
    try:
        result = IngestResult(
            **_svc().update_document_text(doc_id, text, title, tags))
    except KeyError:
        raise ValueError(f"Unknown document: {doc_id}") from None
    logger.info("procure_update_text -> %s", result)
    return result


@mcp.tool()
def procure_delete_document(
    doc_id: Annotated[str, Field(description="Document ID to delete (see procure_list_documents).")],
) -> dict:
    """Delete a document and all its chunks from the library. Use it to remove stale, wrong, or superseded memories. This cannot be undone."""
    logger.info("procure_delete_document doc_id=%r", doc_id)
    if not _svc().delete_document(doc_id):
        raise ValueError(f"Unknown document: {doc_id}")
    return {"deleted": doc_id}


def tool_descriptions() -> list[dict]:
    """Tool name + description for the workspace UI (single source)."""
    out = []
    for name in ("procure_search", "procure_list_documents",
                 "procure_add_text",
                 "procure_update_text", "procure_delete_document"):
        fn = globals().get(name)
        if fn is None:
            continue
        out.append({"name": name, "description": (fn.__doc__ or "").strip()})
    out.append({"name": "procure://documents/{doc_id}",
                "description": "Fetch the full text of a document found via search or listing."})
    out.append({"name": "procure://guide",
                "description": "Agent usage guide — how to search, read documents, and store memories."})
    return out


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
    # The SDK auto-enables DNS-rebinding protection for localhost binds;
    # pass it explicitly so a future default change can't silently open
    # the server to cross-site hosts.
    security = None
    if args.host in ("127.0.0.1", "localhost", "::1"):
        from mcp.server.transport_security import TransportSecuritySettings

        security = TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=["127.0.0.1:*", "localhost:*", "[::1]:*"],
            allowed_origins=["http://127.0.0.1:*", "http://localhost:*",
                             "http://[::1]:*"],
        )
    mcp.run(transport="streamable-http", host=args.host, port=args.port,
            transport_security=security)


if __name__ == "__main__":
    main()
