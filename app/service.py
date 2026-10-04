"""Ingest pipeline + search orchestration."""
from __future__ import annotations

import hashlib
import logging
import os
import threading
import urllib.parse
import uuid
from collections.abc import Callable
from functools import lru_cache

from app import chunking, extract
from app.config import Settings, get_settings
from app.embeddings.local_hash import HashEmbedder
from app.search import (
    CrossEncoderScorer,
    NLIEntailmentScorer,
    SparseStore,
    clear_answerability,
    combine_answerability,
    content_terms,
    coverage_score,
    minmax_norm,
    rrf_fuse,
)
from app.store import MetaStore, normalize_doc_type
from app.vectordb.sqlite_vec import DimensionMismatchError, SqliteVectorStore

__all__ = ["RAGService", "content_hash", "get_service", "DimensionMismatchError"]

logger = logging.getLogger(__name__)

# Candidate pool sizes for hybrid retrieval and reranking.
_RETRIEVE_MULT = 5
_RETRIEVE_MIN = 25
_RERANK_MULT = 3
_RERANK_MIN = 15
# Final blend: normalized fused retrieval score vs answerability.
_W_FUSED = 0.5
_W_ANSWER = 0.5

_TEXT_EXTENSIONS = (".txt", ".md", ".markdown")

ProgressCb = Callable[[int, int, str], None]
CancelCb = Callable[[], bool]


def content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_filename(name: str, default: str = "file") -> str:
    return ("".join(c if c.isalnum() or c in "._- " else "_" for c in name).strip()
            or default)


class RAGService:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        os.makedirs(self.settings.raw_dir, exist_ok=True)
        self.meta = MetaStore(self.settings.meta_db)
        self.embedder = HashEmbedder(self.settings.hash_dim)
        self.vectors = SqliteVectorStore(self.settings.vec_db)
        self.sparse = SparseStore(self.settings.sparse_db)
        self._cross = CrossEncoderScorer(
            self.settings.cross_encoder_model,
            self.settings.cross_encoder_download,
        )
        self._nli = NLIEntailmentScorer(
            self.settings.nli_model,
            self.settings.nli_download,
        )
        # Serializes check-then-insert ingest within this process (job
        # workers + watch loop race otherwise). Cross-process (workspace +
        # MCP subprocess) races can still double-ingest; production callers
        # should dedup by content hash on read.
        self._ingest_lock = threading.RLock()
        self._dim_warned = False
        self._embed_dim_cached: int | None = None
        # One-time backfill for corpora ingested before sparse stats existed.
        if self.sparse.count() == 0 and self.vectors.count() > 0:
            self._backfill_sparse()

    def _backfill_sparse(self) -> None:
        all_chunks = getattr(self.vectors, "all_chunks", None)
        if not callable(all_chunks):
            return
        rows = all_chunks()
        if rows:
            ids = [r[0] for r in rows]
            self.sparse.upsert_chunks(ids, [r[1] for r in rows], [r[2] for r in rows])

    # -- ingest ---------------------------------------------------------
    def ingest_files(self, files: list[tuple[str, bytes]],
                     source: str = "upload", source_uri: str = "",
                     tags: list[str] | None = None,
                     doc_type: str = "document",
                     progress: ProgressCb | None = None,
                     cancelled: CancelCb | None = None) -> list[dict]:
        dtype = normalize_doc_type(doc_type)
        out = []
        for i, (filename, data) in enumerate(files):
            if cancelled is not None and cancelled():
                out.append({"filename": filename, "status": "cancelled"})
                continue
            if progress is not None:
                progress(i, len(files), filename)
            out.append(self._ingest_one(filename, data, source, source_uri,
                                        tags, dtype))
        if progress is not None:
            progress(len(files), len(files), "")
        return out

    def _duplicate_result(self, dup: dict, tags: list[str] | None) -> dict:
        """Dedup response, unioning the caller's tags into the existing doc."""
        merged = sorted(set(self.meta.get_tags(dup["doc_id"]))
                        | {t.strip().lower() for t in (tags or []) if t.strip()})
        applied = self.meta.set_tags(dup["doc_id"], merged)
        return {"doc_id": dup["doc_id"], "filename": dup["filename"],
                "status": "duplicate", "chunk_count": dup["chunk_count"],
                "tags": applied, "doc_type": dup.get("doc_type", "document")}

    @staticmethod
    def _failed(doc_id: str, filename: str, error: str,
                doc_type: str) -> dict:
        return {"doc_id": doc_id, "filename": filename, "status": "failed",
                "error": error, "doc_type": doc_type}

    def _ingest_one(self, filename: str, data: bytes, source: str = "upload",
                    source_uri: str = "", tags: list[str] | None = None,
                    doc_type: str = "document") -> dict:
        dtype = normalize_doc_type(doc_type)
        with self._ingest_lock:
            digest = content_hash(data)
            dup = self.meta.get_by_hash(digest, dtype)
            if dup is not None:
                return self._duplicate_result(dup, tags)
            doc_id = uuid.uuid4().hex[:12]
            safe = _safe_filename(filename)
            raw_path = os.path.join(self.settings.raw_dir, f"{doc_id}_{safe}")
            self.meta.upsert(doc_id, filename, "processing", doc_type=dtype)
            try:
                if not extract.is_supported(filename):
                    raise ValueError(f"Unsupported file type: {filename}")
                with open(raw_path, "wb") as f:
                    f.write(data)
                self.meta.set_raw_path(doc_id, raw_path)
                text = extract.extract_text(
                    raw_path, filename, ocr_mode=self.settings.ocr_mode)
                return self._commit_text(doc_id, filename, text, digest,
                                         source, source_uri, tags, dtype)
            except Exception as e:  # per-file failure must not break batch
                self.meta.update_status(doc_id, "failed", str(e), 0)
                return self._failed(doc_id, filename, str(e), dtype)

    def ingest_text(self, filename: str, text: str,
                    tags: list[str] | None = None,
                    doc_type: str = "document",
                    source: str = "text", source_uri: str = "") -> dict:
        """Ingest raw text (agent transcripts, tool outputs, notes) as a document.

        Same chunk + embed + store pipeline as file upload; the text is also
        saved under raw_dir so the document stays re-ingestible from Library.
        Pass doc_type="memory" to store a cross-agent memory.
        """
        dtype = normalize_doc_type(doc_type)
        with self._ingest_lock:
            digest = content_hash((text or "").encode("utf-8"))
            dup = self.meta.get_by_hash(digest, dtype)
            if dup is not None:
                return self._duplicate_result(dup, tags)
            doc_id = uuid.uuid4().hex[:12]
            safe = _safe_filename(filename, "snippet")
            if os.path.splitext(safe)[1].lower() not in _TEXT_EXTENSIONS:
                safe += ".md"
            filename = (filename.strip() or safe)
            if os.path.splitext(filename)[1].lower() not in _TEXT_EXTENSIONS:
                filename += ".md"
            raw_path = os.path.join(self.settings.raw_dir, f"{doc_id}_{safe}")
            self.meta.upsert(doc_id, filename, "processing", doc_type=dtype)
            try:
                with open(raw_path, "w", encoding="utf-8") as f:
                    f.write(text or "")
                self.meta.set_raw_path(doc_id, raw_path)
                return self._commit_text(doc_id, filename, text, digest,
                                         source, source_uri, tags, dtype)
            except Exception as e:  # surface as failed status
                self.meta.update_status(doc_id, "failed", str(e), 0)
                return self._failed(doc_id, filename, str(e), dtype)

    def _failed_url_result(self, url: str, err: str, dtype: str) -> dict:
        """Failed-URL record, reusing the existing failed doc for this URL.

        Every bad fetch used to mint a fresh persistent "failed" document,
        so one bad URL (or an agent retry loop) cluttered the library
        without bound. The failure stays visible; it just stops multiplying.
        """
        with self._ingest_lock:
            existing = self.meta.get_by_source_uri(url, status="failed")
            if existing is not None:
                self.meta.update_status(existing["doc_id"], "failed", err, 0)
                return self._failed(existing["doc_id"], existing["filename"],
                                    err, dtype)
            doc_id = uuid.uuid4().hex[:12]
            self.meta.upsert(doc_id, url or "url", "failed", 0, err,
                             source="url", source_uri=url, doc_type=dtype)
            return self._failed(doc_id, url, err, dtype)

    def ingest_url(self, url: str, tags: list[str] | None = None,
                   doc_type: str = "document") -> dict:
        """Fetch a URL, extract its article text, and ingest it as a document."""
        from app.webfetch import fetch_url_text

        dtype = normalize_doc_type(doc_type)
        url = (url or "").strip()
        try:
            title, text = fetch_url_text(url, timeout=self.settings.fetch_timeout)
        except Exception as e:  # fetch failure is a failed doc
            return self._failed_url_result(url, str(e), dtype)
        if not text.strip():
            return self._failed_url_result(
                url, "No readable text found on that page", dtype)
        with self._ingest_lock:
            digest = content_hash(text.encode("utf-8"))
            dup = self.meta.get_by_hash(digest, dtype)
            if dup is not None:
                return self._duplicate_result(dup, tags)
            parts = urllib.parse.urlparse(url)
            slug = _safe_filename(title or parts.netloc, "page")[:80] or "page"
            filename = f"{slug}.md"
            doc_id = uuid.uuid4().hex[:12]
            raw_path = os.path.join(self.settings.raw_dir, f"{doc_id}_{slug}.md")
            self.meta.upsert(doc_id, filename, "processing",
                             source="url", source_uri=url, doc_type=dtype)
            try:
                with open(raw_path, "w", encoding="utf-8") as f:
                    f.write(f"# {title}\n\nSource: {url}\n\n{text}")
                self.meta.set_raw_path(doc_id, raw_path)
                return self._commit_text(doc_id, filename, text, digest,
                                         "url", url, tags, dtype)
            except Exception as e:  # surface as failed status
                self.meta.update_status(doc_id, "failed", str(e), 0)
                return self._failed(doc_id, filename, str(e), dtype)

    def ingest_path(self, path: str, source: str = "watch",
                    source_uri: str = "",
                    doc_type: str = "document") -> dict:
        """Ingest a file from disk (folder-watch sync)."""
        with open(path, "rb") as f:
            data = f.read()
        return self._ingest_one(os.path.basename(path), data,
                                source, source_uri or path,
                                None, doc_type)

    def update_document_content(self, doc_id: str, data: bytes,
                                filename: str | None = None) -> dict:
        """Re-ingest new bytes into an existing doc id (watch updates).

        Preserves tags and doc_type; refreshes chunks, hash, and raw copy.
        """
        doc = self.meta.get(doc_id)
        if not doc:
            raise KeyError(doc_id)
        name = filename or doc["filename"]
        dtype = doc.get("doc_type") or "document"
        safe = _safe_filename(name)
        raw_path = os.path.join(self.settings.raw_dir, f"{doc_id}_{safe}")
        for old in os.listdir(self.settings.raw_dir):
            if old.startswith(doc_id + "_") and old != os.path.basename(raw_path):
                try:
                    os.remove(os.path.join(self.settings.raw_dir, old))
                except OSError:
                    pass
        with open(raw_path, "wb") as f:
            f.write(data)
        self.meta.set_raw_path(doc_id, raw_path)
        try:
            text = extract.extract_text(
                raw_path, name, ocr_mode=self.settings.ocr_mode)
            return self._commit_text(doc_id, name, text, content_hash(data),
                                     doc.get("source") or "upload",
                                     doc.get("source_uri") or "",
                                     self.meta.get_tags(doc_id), dtype)
        except Exception as e:  # surface as failed status
            self.meta.update_status(doc_id, "failed", str(e), 0)
            return self._failed(doc_id, name, str(e), dtype)

    def update_document_text(self, doc_id: str, text: str,
                             filename: str | None = None,
                             tags: list[str] | None = None) -> dict:
        """Replace a text document's content in place (memory lifecycle).

        Re-chunks and re-embeds; preserves doc_type/source unless the
        caller passes a new filename/tags. Raises KeyError when unknown.
        """
        doc = self.meta.get(doc_id)
        if not doc:
            raise KeyError(doc_id)
        name = (filename or doc["filename"]).strip()
        if os.path.splitext(name)[1].lower() not in _TEXT_EXTENSIONS:
            name += ".md"
        dtype = normalize_doc_type(doc.get("doc_type") or "document")
        safe = _safe_filename(name, "snippet")
        if os.path.splitext(safe)[1].lower() not in _TEXT_EXTENSIONS:
            safe += ".md"
        raw_path = os.path.join(self.settings.raw_dir, f"{doc_id}_{safe}")
        for old in os.listdir(self.settings.raw_dir):
            if old.startswith(doc_id + "_") and old != os.path.basename(raw_path):
                try:
                    os.remove(os.path.join(self.settings.raw_dir, old))
                except OSError:
                    pass
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(text or "")
        self.meta.set_raw_path(doc_id, raw_path)
        try:
            return self._commit_text(
                doc_id, name, text, content_hash((text or "").encode("utf-8")),
                doc.get("source") or "upload", doc.get("source_uri") or "",
                self.meta.get_tags(doc_id) if tags is None else tags, dtype)
        except Exception as e:  # surface as failed status
            self.meta.update_status(doc_id, "failed", str(e), 0)
            return self._failed(doc_id, name, str(e), dtype)

    def update_document_meta(self, doc_id: str, filename: str | None = None,
                             doc_type: str | None = None) -> dict:
        """Rename a document and/or change its type. Raises KeyError."""
        if not self.meta.get(doc_id):
            raise KeyError(doc_id)
        name = filename.strip() if filename is not None else None
        if name == "":
            raise ValueError("filename must not be empty")
        updated = self.meta.update_meta(doc_id, name, doc_type)
        assert updated is not None
        return {**updated, "tags": self.meta.get_tags(doc_id)}

    def _commit_text(self, doc_id: str, filename: str, text: str,
                     digest: str = "", source: str = "upload",
                     source_uri: str = "",
                     tags: list[str] | None = None,
                     doc_type: str = "document") -> dict:
        dtype = normalize_doc_type(doc_type)
        chunks = chunking.split_text(
            text, self.settings.chunk_size, self.settings.chunk_overlap
        )
        if not chunks:
            raise ValueError("No text extracted")
        embs = self.embedder.embed(chunks)
        ids = [f"{doc_id}:{i}" for i in range(len(chunks))]
        # Delete first: without this, shrinking content leaves stale chunks
        # ({doc_id}:n..) behind in both stores, still served by search.
        self.vectors.delete_by_doc(doc_id)
        self.sparse.delete_by_doc(doc_id)
        self.vectors.upsert(ids, [doc_id] * len(chunks), embs, chunks)
        self.sparse.upsert_chunks(ids, [doc_id] * len(chunks), chunks)
        self.meta.upsert(doc_id, filename, "ready", len(chunks), "",
                         digest, source, source_uri, dtype)
        applied = self.meta.set_tags(doc_id, tags or [])
        self._write_text_cache(doc_id, text)
        return {"doc_id": doc_id, "filename": filename, "status": "ready",
                "chunk_count": len(chunks), "tags": applied,
                "doc_type": dtype}

    # -- extracted-text cache -------------------------------------------
    def _text_cache_path(self, doc_id: str) -> str:
        return os.path.join(self.settings.raw_dir, f"{doc_id}.extracted.txt")

    def _write_text_cache(self, doc_id: str, text: str) -> None:
        try:
            with open(self._text_cache_path(doc_id), "w", encoding="utf-8") as f:
                f.write(text or "")
        except OSError:
            pass  # cache is best-effort; extraction stays the fallback

    def _read_text_cache(self, doc_id: str) -> str | None:
        try:
            with open(self._text_cache_path(doc_id), encoding="utf-8") as f:
                return f.read()
        except OSError:
            return None

    # -- library --------------------------------------------------------
    def list_documents(self, doc_type: str | None = None,
                       source: str | None = None,
                       query: str | None = None,
                       limit: int | None = None, offset: int = 0,
                       sort: str = "newest") -> list[dict]:
        return self.meta.list(doc_type, source, query, limit, offset, sort)

    def count_documents(self, doc_type: str | None = None,
                        source: str | None = None,
                        query: str | None = None) -> int:
        return self.meta.count_matching(doc_type, source, query)

    def library_version(self) -> dict:
        """Cheap revision fingerprint; never touches the vector stores."""
        return self.meta.library_version()

    def get_document(self, doc_id: str) -> dict:
        """Full document record: metadata plus complete extracted text.

        The text comes from the cache written at ingest; when it is missing
        (older corpora) the raw file is re-extracted once and cached.
        Raises KeyError for unknown doc ids.
        """
        doc = self.meta.get(doc_id)
        if not doc:
            raise KeyError(doc_id)
        text = self._read_text_cache(doc_id)
        if text is None:
            text = ""
            raw_path = self._raw_path(doc_id)
            if raw_path is not None:
                try:
                    text = extract.extract_text(
                        raw_path, doc["filename"],
                        ocr_mode=self.settings.ocr_mode)
                except Exception:  # fall back to stored chunks
                    text = ""
            if text and doc.get("source") == "url":
                text = self._strip_url_header(text, doc)
            if text:
                self._write_text_cache(doc_id, text)
        if not text:
            rows = self.vectors.get_by_doc(doc_id)
            text = "\n\n".join(t for _, t in rows if t)
        return {**doc, "tags": self.meta.get_tags(doc_id), "text": text}

    @staticmethod
    def _strip_url_header(text: str, doc: dict) -> str:
        """Drop the "# title / Source: url" header from legacy URL raws.

        The raw copy of a URL doc starts with a 3-line header that was
        never part of the indexed text; cached/indexed reads exclude it.
        """
        lines = text.split("\n")
        uri = doc.get("source_uri") or ""
        if (len(lines) >= 3 and lines[0].startswith("# ")
                and lines[1] == f"Source: {uri}"):
            return "\n".join(lines[3:]).lstrip("\n")
        return text

    def _raw_path(self, doc_id: str) -> str | None:
        doc = self.meta.get(doc_id)
        if doc and doc.get("raw_path"):
            return doc["raw_path"]
        # Legacy rows predate the raw_path column: scan once, then pin it.
        try:
            names = os.listdir(self.settings.raw_dir)
        except OSError:
            return None
        found = next(
            (os.path.join(self.settings.raw_dir, n) for n in names
             if n.startswith(doc_id + "_") and not n.endswith(".extracted.txt")),
            None,
        )
        if found is not None:
            self.meta.set_raw_path(doc_id, found)
        return found

    # -- tags -----------------------------------------------------------
    def set_tags(self, doc_id: str, tags: list[str]) -> dict:
        if not self.meta.get(doc_id):
            raise KeyError(doc_id)
        return {"doc_id": doc_id, "tags": self.meta.set_tags(doc_id, tags)}

    def list_tags(self) -> list[dict]:
        return self.meta.all_tags()

    # -- runtime settings -----------------------------------------------
    def update_settings(self, search: str | None = None,
                        rerank: str | None = None) -> dict:
        """Change retrieval settings (persisted; the MCP server reads the
        persisted values when it is (re)started).

        Raises ValueError on unknown values.
        """
        if search is not None:
            mode = search.strip().lower()
            if mode not in ("hybrid", "dense"):
                raise ValueError(f"Unknown search mode: {search!r} (want hybrid|dense)")
            self.settings.search_mode = mode
        if rerank is not None:
            flag = rerank.strip().lower()
            if flag not in ("on", "off"):
                raise ValueError(f"Unknown rerank flag: {rerank!r} (want on|off)")
            self.settings.rerank = flag
        if search is not None or rerank is not None:
            self.settings.save_overrides()
        return {"search": self.settings.search_mode, "rerank": self.settings.rerank}

    def set_mcp_autostart(self, enabled: bool) -> dict:
        self.settings.mcp_autostart = bool(enabled)
        self.settings.save_overrides()
        return {"mcp_autostart": self.settings.mcp_autostart}

    def delete_document(self, doc_id: str) -> bool:
        doc = self.meta.get(doc_id)
        if not doc:
            return False
        self.vectors.delete_by_doc(doc_id)
        self.sparse.delete_by_doc(doc_id)
        raw_path = doc.get("raw_path") or ""
        if raw_path:
            try:
                os.remove(raw_path)
            except OSError:
                pass
        try:
            os.remove(self._text_cache_path(doc_id))
        except OSError:
            pass
        # Legacy rows may carry raw files the column doesn't know about.
        try:
            names = os.listdir(self.settings.raw_dir)
        except OSError:
            names = []
        for name in names:
            if name.startswith(doc_id + "_"):
                try:
                    os.remove(os.path.join(self.settings.raw_dir, name))
                except OSError:
                    pass
        self.meta.delete(doc_id)
        return True

    def mark_missing(self, doc_id: str, missing: bool) -> None:
        self.meta.mark_missing(doc_id, missing)

    def reingest_document(self, doc_id: str) -> dict:
        doc = self.meta.get(doc_id)
        if not doc:
            raise KeyError(doc_id)
        dtype = doc.get("doc_type") or "document"
        raw_path = self._raw_path(doc_id)
        if raw_path is None:
            raise FileNotFoundError(f"Raw file for {doc_id} is gone; re-upload it")
        self.meta.update_status(doc_id, "processing", "")
        try:
            text = extract.extract_text(
                raw_path, doc["filename"], ocr_mode=self.settings.ocr_mode)
            with open(raw_path, "rb") as f:
                digest = content_hash(f.read())
            return self._commit_text(
                doc_id, doc["filename"], text, digest,
                doc.get("source") or "upload", doc.get("source_uri") or "",
                self.meta.get_tags(doc_id), dtype)
        except Exception as e:  # surface as failed status
            self.meta.update_status(doc_id, "failed", str(e), 0)
            return self._failed(doc_id, doc["filename"], str(e), dtype)

    # -- embedding-dimension health --------------------------------------
    def _current_dim(self) -> int | None:
        if self._embed_dim_cached is None:
            try:
                self._embed_dim_cached = int(self.embedder.dim)
            except Exception:  # embedder misconfigured: dim unknown
                return None
        return self._embed_dim_cached

    def dimension_status(self) -> dict:
        """Compare stored vector dims with the active embedder.

        Changing PROCURE_HASH_DIM leaves old vectors behind; dense
        search skips them, so report it instead of failing silently.
        """
        stored = getattr(self.vectors, "stored_dim", lambda: None)()
        current = self._current_dim()
        if stored is None or current is None:
            return {"stored_dim": stored, "embed_dim": current,
                    "dense_ok": True, "warning": ""}
        ok = stored == current
        return {
            "stored_dim": stored,
            "embed_dim": current,
            "dense_ok": ok,
            "warning": ("" if ok else
                        f"Stored vectors are dim {stored} but the embedder is "
                        f"dim {current}: dense search skips old chunks. "
                        "Re-ingest affected documents."),
        }

    # -- export / import -------------------------------------------------
    def export_library(self) -> dict:
        """Logical backup: metadata + extracted text for every readable doc."""
        docs = []
        for meta in self.meta.list():
            try:
                full = self.get_document(meta["doc_id"])
            except KeyError:
                continue
            if not (full.get("text") or "").strip():
                continue
            docs.append({
                "filename": full["filename"],
                "text": full["text"],
                "tags": full.get("tags", []),
                "doc_type": full.get("doc_type", "document"),
                "source": full.get("source", "upload"),
                "source_uri": full.get("source_uri", ""),
                "created_at": full.get("created_at", ""),
            })
        return {"version": 1, "documents": docs}

    def import_library(self, payload: dict,
                       progress: ProgressCb | None = None,
                       cancelled: CancelCb | None = None) -> list[dict]:
        """Restore an export_library() payload. Returns per-doc results."""
        items = payload.get("documents", []) if isinstance(payload, dict) else []
        out = []
        for i, item in enumerate(items):
            if cancelled is not None and cancelled():
                out.append({"filename": item.get("filename", "?"),
                            "status": "cancelled"})
                continue
            if progress is not None:
                progress(i, len(items), item.get("filename", ""))
            source = item.get("source") or "text"
            if source not in ("upload", "url", "text"):
                source = "text"
            try:
                out.append(self.ingest_text(
                    item.get("filename", "import"), item.get("text", ""),
                    item.get("tags", []), item.get("doc_type", "document"),
                    source, item.get("source_uri", "")))
            except Exception as e:  # per-doc failure must not break batch
                out.append({"filename": item.get("filename", "?"),
                            "status": "failed", "error": str(e)})
        if progress is not None:
            progress(len(items), len(items), "")
        return out

    # -- search ---------------------------------------------------------
    def search(self, query: str, top_k: int = 5,
               doc_ids: list[str] | None = None,
               tags: list[str] | None = None,
               source: str | None = None,
               doc_type: str | None = None,
               since: str | None = None) -> list[dict]:
        if not query.strip():
            return []
        doc_ids = self._resolve_scope(doc_ids, tags, source, doc_type, since)
        # A filter that matches nothing must return nothing (not everything).
        if (doc_ids is not None) and not doc_ids:
            return []
        hybrid = self.settings.hybrid
        rerank_on = self.settings.rerank_enabled
        depth = max(top_k * _RETRIEVE_MULT, _RETRIEVE_MIN)
        q = self.embedder.embed([query])[0]
        dense_hits = self.vectors.search(q, depth if (hybrid or rerank_on) else top_k,
                                         doc_ids)
        skipped = getattr(self.vectors, "last_skipped", 0) or 0
        if skipped and not self._dim_warned:
            self._dim_warned = True
            logger.warning(
                "dense search skipped %d chunk(s) with a stale embedding "
                "dimension (backend was switched?)", skipped)
        dense_by_id = {h.chunk_id: h for h in dense_hits}

        sparse_by_id: dict[str, float] = {}
        sparse_text: dict[str, tuple[str, str]] = {}
        if hybrid:
            for s in self.sparse.search(query, depth, doc_ids):
                sparse_by_id[s.chunk_id] = s.score
                sparse_text[s.chunk_id] = (s.doc_id, s.text)

        if hybrid:
            fused = rrf_fuse(
                [[h.chunk_id for h in dense_hits], list(sparse_by_id.keys())])
            ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
        else:
            fused = {h.chunk_id: h.score for h in dense_hits}
            ordered = [(h.chunk_id, h.score) for h in dense_hits]
        fused_rank = {cid: i + 1 for i, (cid, _) in enumerate(ordered)}

        pool = [cid for cid, _ in ordered[:max(top_k * _RERANK_MULT, _RERANK_MIN)]]

        texts = {
            cid: (dense_by_id[cid].text if cid in dense_by_id
                  else sparse_text[cid][1])
            for cid in pool if cid in dense_by_id or cid in sparse_text
        }
        doc_of = {
            cid: (dense_by_id[cid].doc_id if cid in dense_by_id
                  else sparse_text[cid][0])
            for cid in texts
        }

        answer: dict[str, float] = {}
        entail: dict[str, float] = {}
        if rerank_on and texts:
            pool_texts = [texts[cid] for cid in texts]
            cross = self._cross.score(query, pool_texts)
            # NLI only runs when the cross-encoder did: the fallback path
            # never consumes entailment, so running it alone is pure waste.
            ent = self._nli.score(query, pool_texts) if cross is not None else None
            if cross is not None and ent is not None:
                # CLEAR inference (mode "all"): sigmoid(relevance) + alpha * entailment.
                for cid, a, e in zip(texts, clear_answerability(
                        cross, ent, self.settings.alpha_nli), ent):
                    answer[cid] = a
                    entail[cid] = e
            else:
                idf = self.sparse.idf_map(content_terms(query))
                cov = [coverage_score(query, texts[cid], idf) for cid in texts]
                for cid, a in zip(texts, combine_answerability(cross, cov)):
                    answer[cid] = a

        if rerank_on and texts:
            fnorm = minmax_norm({cid: fused[cid] for cid in texts})
            final = {cid: _W_FUSED * fnorm[cid] + _W_ANSWER * answer[cid]
                     for cid in texts}
            ranked = sorted(final.items(), key=lambda kv: (-kv[1], fused_rank[kv[0]]))
        else:
            ranked = [(cid, fused[cid]) for cid in pool if cid in texts]

        # Hydrate metadata for the returned hits only (never the whole library).
        meta = self.meta.get_many([doc_of[cid] for cid, _ in ranked[:top_k]])
        out = []
        for cid, score in ranked[:top_k]:
            did = doc_of[cid]
            doc = meta.get(did, {})
            out.append({
                "chunk_id": cid, "doc_id": did,
                "filename": doc.get("filename", did),
                "doc_type": doc.get("doc_type", "document"),
                "text": texts[cid], "score": round(float(score), 4),
                "sparse_score": round(float(sparse_by_id.get(cid, 0.0)), 4),
                "dense_score": round(float(
                    dense_by_id[cid].score if cid in dense_by_id else 0.0), 4),
                "fused_rank": fused_rank[cid],
                "answerability": round(float(answer.get(cid, 0.0)), 4),
                "entailment": round(float(entail.get(cid, 0.0)), 4),
            })
        return out

    def _resolve_scope(self, doc_ids: list[str] | None,
                       tags: list[str] | None,
                       source: str | None,
                       doc_type: str | None = None,
                       since: str | None = None) -> list[str] | None:
        """Intersect explicit doc ids with tag/source/type/recency filters.

        Tags are document-level metadata; chunks inherit their document's
        tags at query time, so tag filtering resolves to a doc-id
        pre-filter applied to both retrieval channels. Returns None when
        no filter was given (search everything).
        """
        sets: list[set[str]] = []
        if doc_ids:
            sets.append(set(doc_ids))
        if tags:
            sets.append(set(self.meta.doc_ids_for_tags(tags)))
        if source:
            sets.append(set(self.meta.doc_ids_for_source(source.strip().lower())))
        dtype = normalize_doc_type(doc_type, allow_empty=True)
        if dtype is not None:
            sets.append(set(self.meta.doc_ids_for_doc_type(dtype)))
        if since:
            sets.append(set(self.meta.doc_ids_created_since(since)))
        if not sets:
            return None
        out = sets[0]
        for s in sets[1:]:
            out &= s
        return sorted(out)


@lru_cache(maxsize=1)
def get_service() -> RAGService:
    return RAGService()
