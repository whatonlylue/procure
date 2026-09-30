"""SQLite metadata store for documents, tags, and watched folders."""
from __future__ import annotations

import os
import re
import sqlite3
from datetime import datetime, timezone

_TAG_CLEAN = re.compile(r"\s+")


def normalize_tag(tag: str) -> str:
    """Lowercase, collapse whitespace, drop empties. Max 64 chars."""
    return _TAG_CLEAN.sub(" ", (tag or "").strip().lower())[:64]


DOC_TYPE_DOCUMENT = "document"
DOC_TYPE_MEMORY = "memory"

_MEMORY_ALIASES = frozenset({
    "memory", "memories", "agent-memory", "agent-memories",
    "agent_memory", "agent_memories", "agentmemory", "agentmemories",
})
_DOCUMENT_ALIASES = frozenset({
    "document", "documents", "doc", "docs",
})


def normalize_doc_type(value: str | None,
                       allow_empty: bool = False) -> str | None:
    """Canonical doc_type: "document" (default) or "memory" (agent memory).

    Accepts aliases such as "agent memories" / "agent-memory" for memory.
    With allow_empty (search/list filters), None/"" means no filter.
    Raises ValueError on unknown values.
    """
    if value is None or not str(value).strip():
        return None if allow_empty else DOC_TYPE_DOCUMENT
    clean = _TAG_CLEAN.sub(" ", str(value).strip().lower())
    compact = clean.replace(" ", "").replace("-", "").replace("_", "")
    if clean in _MEMORY_ALIASES or compact in ("memory", "memories",
                                              "agentmemory", "agentmemories"):
        return DOC_TYPE_MEMORY
    if clean in _DOCUMENT_ALIASES or compact in ("document", "documents",
                                                "doc", "docs"):
        return DOC_TYPE_DOCUMENT
    raise ValueError(
        f"Unknown doc_type: {value!r} (want document|memory)")


class MetaStore:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        with self._connect() as c:
            c.execute(
                """CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    error TEXT DEFAULT '',
                    created_at TEXT NOT NULL
                )"""
            )
            self._migrate(c)
            c.execute(
                """CREATE TABLE IF NOT EXISTS doc_tags (
                    doc_id TEXT NOT NULL,
                    tag TEXT NOT NULL,
                    PRIMARY KEY (doc_id, tag)
                )"""
            )
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_doc_tags_tag ON doc_tags(tag)"
            )
            c.execute(
                """CREATE TABLE IF NOT EXISTS watch_folders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    path TEXT NOT NULL UNIQUE,
                    recursive INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL
                )"""
            )
            c.execute(
                """CREATE TABLE IF NOT EXISTS watched_files (
                    path TEXT PRIMARY KEY,
                    mtime REAL NOT NULL DEFAULT 0,
                    size INTEGER NOT NULL DEFAULT 0,
                    content_hash TEXT NOT NULL DEFAULT '',
                    doc_id TEXT NOT NULL DEFAULT ''
                )"""
            )

    def _migrate(self, c: sqlite3.Connection) -> None:
        cols = {r[1] for r in c.execute("PRAGMA table_info(documents)").fetchall()}
        if "content_hash" not in cols:
            c.execute("ALTER TABLE documents ADD COLUMN content_hash TEXT DEFAULT ''")
        if "source" not in cols:
            c.execute("ALTER TABLE documents ADD COLUMN source TEXT DEFAULT 'upload'")
        if "source_uri" not in cols:
            c.execute("ALTER TABLE documents ADD COLUMN source_uri TEXT DEFAULT ''")
        if "doc_type" not in cols:
            c.execute("ALTER TABLE documents ADD COLUMN doc_type TEXT DEFAULT 'document'")
        c.execute(
            "UPDATE documents SET doc_type='document' "
            "WHERE doc_type IS NULL OR doc_type=''"
        )
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(content_hash)"
        )
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source)"
        )
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_documents_doc_type ON documents(doc_type)"
        )

    def _connect(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.path)
        c.row_factory = sqlite3.Row
        return c

    # -- documents ------------------------------------------------------
    def upsert(self, doc_id: str, filename: str, status: str,
               chunk_count: int = 0, error: str = "",
               content_hash: str = "", source: str = "upload",
               source_uri: str = "",
               doc_type: str = DOC_TYPE_DOCUMENT) -> None:
        now = datetime.now(timezone.utc).isoformat()
        doc_type = normalize_doc_type(doc_type)
        with self._connect() as c:
            c.execute(
                """INSERT INTO documents
                   (doc_id, filename, status, chunk_count, error, created_at,
                    content_hash, source, source_uri, doc_type)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(doc_id) DO UPDATE SET
                     filename=excluded.filename, status=excluded.status,
                     chunk_count=excluded.chunk_count, error=excluded.error,
                     content_hash=excluded.content_hash, source=excluded.source,
                     source_uri=excluded.source_uri, doc_type=excluded.doc_type""",
                (doc_id, filename, status, chunk_count, error, now,
                 content_hash, source, source_uri, doc_type),
            )

    def get(self, doc_id: str) -> dict | None:
        with self._connect() as c:
            row = c.execute("SELECT * FROM documents WHERE doc_id=?", (doc_id,)).fetchone()
        if not row:
            return None
        doc = dict(row)
        doc.setdefault("doc_type", DOC_TYPE_DOCUMENT)
        return doc

    def get_by_hash(self, content_hash: str) -> dict | None:
        """Most recent live document with this content hash (dedup lookup)."""
        if not content_hash:
            return None
        with self._connect() as c:
            row = c.execute(
                """SELECT * FROM documents WHERE content_hash=? AND status != 'missing'
                   ORDER BY created_at DESC LIMIT 1""",
                (content_hash,),
            ).fetchone()
        if not row:
            return None
        doc = dict(row)
        doc.setdefault("doc_type", DOC_TYPE_DOCUMENT)
        return doc

    def list(self, doc_type: str | None = None) -> list[dict]:
        dtype = normalize_doc_type(doc_type, allow_empty=True)
        with self._connect() as c:
            if dtype is None:
                rows = c.execute(
                    "SELECT * FROM documents ORDER BY created_at DESC").fetchall()
            else:
                rows = c.execute(
                    "SELECT * FROM documents WHERE doc_type=? "
                    "ORDER BY created_at DESC", (dtype,)).fetchall()
        docs = [dict(r) for r in rows]
        tags = self._tags_for([d["doc_id"] for d in docs])
        for d in docs:
            d["tags"] = tags.get(d["doc_id"], [])
            d.setdefault("doc_type", DOC_TYPE_DOCUMENT)
        return docs

    def delete(self, doc_id: str) -> bool:
        with self._connect() as c:
            c.execute("DELETE FROM doc_tags WHERE doc_id=?", (doc_id,))
            cur = c.execute("DELETE FROM documents WHERE doc_id=?", (doc_id,))
        return cur.rowcount > 0

    def mark_missing(self, doc_id: str, missing: bool) -> None:
        with self._connect() as c:
            if missing:
                c.execute(
                    "UPDATE documents SET status='missing', "
                    "error='source file removed' WHERE doc_id=?",
                    (doc_id,),
                )
            else:
                c.execute(
                    "UPDATE documents SET status='ready', error='' "
                    "WHERE doc_id=? AND status='missing'",
                    (doc_id,),
                )

    # -- tags -----------------------------------------------------------
    def set_tags(self, doc_id: str, tags: list[str]) -> list[str]:
        clean = sorted({t for t in (normalize_tag(t) for t in tags) if t})
        with self._connect() as c:
            c.execute("DELETE FROM doc_tags WHERE doc_id=?", (doc_id,))
            c.executemany(
                "INSERT INTO doc_tags (doc_id, tag) VALUES (?, ?)",
                [(doc_id, t) for t in clean],
            )
        return clean

    def get_tags(self, doc_id: str) -> list[str]:
        return self._tags_for([doc_id]).get(doc_id, [])

    def _tags_for(self, doc_ids: list[str]) -> dict[str, list[str]]:
        if not doc_ids:
            return {}
        with self._connect() as c:
            ph = ",".join("?" for _ in doc_ids)
            rows = c.execute(
                f"SELECT doc_id, tag FROM doc_tags WHERE doc_id IN ({ph}) "
                "ORDER BY tag",
                doc_ids,
            ).fetchall()
        out: dict[str, list[str]] = {}
        for did, tag in rows:
            out.setdefault(did, []).append(tag)
        return out

    def all_tags(self) -> list[dict]:
        with self._connect() as c:
            rows = c.execute(
                "SELECT tag, COUNT(*) AS n FROM doc_tags "
                "GROUP BY tag ORDER BY n DESC, tag"
            ).fetchall()
        return [{"tag": r[0], "count": r[1]} for r in rows]

    def doc_ids_for_tags(self, tags: list[str]) -> list[str]:
        """Doc ids carrying ALL of the given tags (AND semantics)."""
        clean = [t for t in (normalize_tag(t) for t in tags) if t]
        if not clean:
            return []
        with self._connect() as c:
            ph = ",".join("?" for _ in clean)
            rows = c.execute(
                f"""SELECT doc_id FROM doc_tags WHERE tag IN ({ph})
                    GROUP BY doc_id HAVING COUNT(DISTINCT tag)=?""",
                (*clean, len(clean)),
            ).fetchall()
        return [r[0] for r in rows]

    def doc_ids_for_source(self, source: str) -> list[str]:
        with self._connect() as c:
            rows = c.execute(
                "SELECT doc_id FROM documents WHERE source=?", (source,)
            ).fetchall()
        return [r[0] for r in rows]

    def doc_ids_for_doc_type(self, doc_type: str) -> list[str]:
        dtype = normalize_doc_type(doc_type)
        with self._connect() as c:
            rows = c.execute(
                "SELECT doc_id FROM documents WHERE doc_type=?", (dtype,)
            ).fetchall()
        return [r[0] for r in rows]

    # -- watched folders ------------------------------------------------
    def add_watch(self, path: str, recursive: bool = True) -> dict:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as c:
            try:
                cur = c.execute(
                    "INSERT INTO watch_folders (path, recursive, created_at) "
                    "VALUES (?, ?, ?)",
                    (path, 1 if recursive else 0, now),
                )
            except sqlite3.IntegrityError:
                row = c.execute(
                    "SELECT * FROM watch_folders WHERE path=?", (path,)
                ).fetchone()
                return dict(row)
            row = c.execute(
                "SELECT * FROM watch_folders WHERE id=?", (cur.lastrowid,)
            ).fetchone()
        return dict(row)

    def list_watches(self) -> list[dict]:
        with self._connect() as c:
            rows = c.execute(
                "SELECT * FROM watch_folders ORDER BY created_at").fetchall()
        return [dict(r) for r in rows]

    def remove_watch(self, watch_id: int) -> bool:
        with self._connect() as c:
            cur = c.execute(
                "DELETE FROM watch_folders WHERE id=?", (watch_id,))
        return cur.rowcount > 0

    def get_watched_file(self, path: str) -> dict | None:
        with self._connect() as c:
            row = c.execute(
                "SELECT * FROM watched_files WHERE path=?", (path,)).fetchone()
        return dict(row) if row else None

    def set_watched_file(self, path: str, mtime: float, size: int,
                         content_hash: str, doc_id: str) -> None:
        with self._connect() as c:
            c.execute(
                """INSERT INTO watched_files (path, mtime, size, content_hash, doc_id)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(path) DO UPDATE SET
                     mtime=excluded.mtime, size=excluded.size,
                     content_hash=excluded.content_hash, doc_id=excluded.doc_id""",
                (path, mtime, size, content_hash, doc_id),
            )

    def drop_watched_file(self, path: str) -> None:
        with self._connect() as c:
            c.execute("DELETE FROM watched_files WHERE path=?", (path,))

    def watched_files_for(self, prefix: str) -> list[dict]:
        with self._connect() as c:
            rows = c.execute(
                "SELECT * FROM watched_files WHERE path=? OR path LIKE ?",
                (prefix, prefix.rstrip(os.sep) + os.sep + "%"),
            ).fetchall()
        return [dict(r) for r in rows]
