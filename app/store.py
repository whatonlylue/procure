"""SQLite metadata store for documents, tags, and watched folders."""
from __future__ import annotations

import contextlib
import os
import re
import sqlite3
from collections.abc import Iterator
from datetime import datetime, timezone

_TAG_CLEAN = re.compile(r"\s+")

# SQLite caps bound variables per statement (999 on older builds); every
# IN (...) list in this module is sent in chunks of at most this size.
_IN_CHUNK = 500


def _batched(seq: list[str], n: int = _IN_CHUNK) -> Iterator[list[str]]:
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def normalize_tag(tag: str) -> str:
    """Lowercase, collapse whitespace. Max 64 chars; empties become ""."""
    return _TAG_CLEAN.sub(" ", (tag or "").strip().lower())[:64]


DOC_TYPE_DOCUMENT = "document"
DOC_TYPE_MEMORY = "memory"


def normalize_doc_type(value: str | None,
                       allow_empty: bool = False) -> str | None:
    """Canonical doc_type: "document" (default) or "memory" (agent memory).

    Accepts aliases such as "agent memories" / "agent-memory" for memory.
    With allow_empty (search/list filters), None/"" means no filter.
    Raises ValueError on unknown values.
    """
    if value is None or not str(value).strip():
        return None if allow_empty else DOC_TYPE_DOCUMENT
    compact = _TAG_CLEAN.sub("", str(value).strip().lower())
    compact = compact.replace("-", "").replace("_", "")
    if compact in ("memory", "memories", "agentmemory", "agentmemories"):
        return DOC_TYPE_MEMORY
    if compact in ("document", "documents", "doc", "docs"):
        return DOC_TYPE_DOCUMENT
    raise ValueError(
        f"Unknown doc_type: {value!r} (want document|memory)")


class MetaStore:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        with self._session() as c:
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
            self._migrate(c)

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
        if "raw_path" not in cols:
            c.execute("ALTER TABLE documents ADD COLUMN raw_path TEXT DEFAULT ''")
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
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_documents_created ON documents(created_at)"
        )
        wcols = {r[1] for r in c.execute("PRAGMA table_info(watch_folders)").fetchall()}
        if "last_sync" not in wcols:
            c.execute("ALTER TABLE watch_folders ADD COLUMN last_sync TEXT DEFAULT ''")
        if "last_error" not in wcols:
            c.execute("ALTER TABLE watch_folders ADD COLUMN last_error TEXT DEFAULT ''")

    def _connect(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.path, timeout=10.0)
        c.row_factory = sqlite3.Row
        # The workspace and the MCP subprocess write the same DBs: WAL lets
        # readers proceed during writes, and busy_timeout turns a locked
        # moment into a short wait instead of an OperationalError.
        c.execute("PRAGMA journal_mode=WAL").fetchone()
        c.execute("PRAGMA busy_timeout=5000")
        c.execute("PRAGMA synchronous=NORMAL")
        return c

    @contextlib.contextmanager
    def _session(self) -> Iterator[sqlite3.Connection]:
        """A connection that commits (or rolls back) and always closes."""
        c = self._connect()
        try:
            with c:
                yield c
        finally:
            c.close()

    # -- documents ------------------------------------------------------
    def upsert(self, doc_id: str, filename: str, status: str,
               chunk_count: int = 0, error: str = "",
               content_hash: str = "", source: str = "upload",
               source_uri: str = "",
               doc_type: str = DOC_TYPE_DOCUMENT) -> None:
        now = datetime.now(timezone.utc).isoformat()
        doc_type = normalize_doc_type(doc_type)
        with self._session() as c:
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

    def update_status(self, doc_id: str, status: str, error: str = "",
                      chunk_count: int | None = None) -> None:
        """Status-only update: preserves hash/source/uri/type/filename.

        Use for "processing" markers and failure records on docs that
        already carry provenance; a bare upsert() would wipe those fields
        back to their defaults.
        """
        with self._session() as c:
            if chunk_count is None:
                c.execute(
                    "UPDATE documents SET status=?, error=? WHERE doc_id=?",
                    (status, error, doc_id),
                )
            else:
                c.execute(
                    "UPDATE documents SET status=?, error=?, chunk_count=? "
                    "WHERE doc_id=?",
                    (status, error, chunk_count, doc_id),
                )

    def update_meta(self, doc_id: str, filename: str | None = None,
                    doc_type: str | None = None) -> dict | None:
        """Rename a document and/or change its type. Returns the row, if any."""
        if filename is None and doc_type is None:
            return self.get(doc_id)
        dtype = (normalize_doc_type(doc_type) if doc_type is not None else None)
        with self._session() as c:
            if filename is not None:
                c.execute("UPDATE documents SET filename=? WHERE doc_id=?",
                          (filename, doc_id))
            if dtype is not None:
                c.execute("UPDATE documents SET doc_type=? WHERE doc_id=?",
                          (dtype, doc_id))
        return self.get(doc_id)

    def set_raw_path(self, doc_id: str, raw_path: str) -> None:
        with self._session() as c:
            c.execute("UPDATE documents SET raw_path=? WHERE doc_id=?",
                      (raw_path, doc_id))

    def get(self, doc_id: str) -> dict | None:
        with self._session() as c:
            row = c.execute("SELECT * FROM documents WHERE doc_id=?", (doc_id,)).fetchone()
        if not row:
            return None
        return dict(row)

    def get_many(self, doc_ids: list[str]) -> dict[str, dict]:
        """Fetch just these documents (search hydrates hits only, not all)."""
        out: dict[str, dict] = {}
        ids = list(dict.fromkeys(doc_ids))
        if not ids:
            return out
        with self._session() as c:
            for batch in _batched(ids):
                ph = ",".join("?" for _ in batch)
                for row in c.execute(
                        f"SELECT * FROM documents WHERE doc_id IN ({ph})",
                        batch).fetchall():
                    out[row["doc_id"]] = dict(row)
        return out

    def get_by_hash(self, content_hash: str,
                    doc_type: str | None = None) -> dict | None:
        """Most recent live document with this hash (dedup lookup).

        Scoped by doc_type when given: the same text may exist once as a
        document and once as a memory, since they are separate logical docs.
        """
        if not content_hash:
            return None
        with self._session() as c:
            if doc_type is None:
                row = c.execute(
                    """SELECT * FROM documents WHERE content_hash=? AND status != 'missing'
                       ORDER BY created_at DESC LIMIT 1""",
                    (content_hash,),
                ).fetchone()
            else:
                row = c.execute(
                    """SELECT * FROM documents
                       WHERE content_hash=? AND doc_type=? AND status != 'missing'
                       ORDER BY created_at DESC LIMIT 1""",
                    (content_hash, normalize_doc_type(doc_type)),
                ).fetchone()
        if not row:
            return None
        return dict(row)

    def get_by_source_uri(self, source_uri: str,
                          status: str | None = None) -> dict | None:
        """Most recent document ingested from this URI (URL retry reuse)."""
        if not source_uri:
            return None
        with self._session() as c:
            if status is None:
                row = c.execute(
                    """SELECT * FROM documents WHERE source_uri=?
                       ORDER BY created_at DESC LIMIT 1""",
                    (source_uri,)).fetchone()
            else:
                row = c.execute(
                    """SELECT * FROM documents WHERE source_uri=? AND status=?
                       ORDER BY created_at DESC LIMIT 1""",
                    (source_uri, status)).fetchone()
        return dict(row) if row else None

    _SORTS = {
        "newest": "created_at DESC, doc_id DESC",
        "oldest": "created_at ASC, doc_id ASC",
        "name": "filename COLLATE NOCASE ASC, doc_id ASC",
    }

    def _filter_sql(self, doc_type: str | None, source: str | None,
                    query: str | None) -> tuple[str, list[str]]:
        dtype = normalize_doc_type(doc_type, allow_empty=True)
        clauses: list[str] = []
        params: list[str] = []
        if dtype is not None:
            clauses.append("doc_type=?")
            params.append(dtype)
        if source:
            clauses.append("source=?")
            params.append(source.strip().lower())
        if query and query.strip():
            clauses.append("(filename LIKE ? ESCAPE '\\' OR doc_id=?)")
            params.extend([f"%{_escape_like(query.strip())}%", query.strip()])
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        return where, params

    def list(self, doc_type: str | None = None,
             source: str | None = None,
             query: str | None = None,
             limit: int | None = None, offset: int = 0,
             sort: str = "newest") -> list[dict]:
        order = self._SORTS.get((sort or "newest").strip().lower(),
                              self._SORTS["newest"])
        where, params = self._filter_sql(doc_type, source, query)
        sql = f"SELECT * FROM documents {where} ORDER BY {order}".rstrip()
        if limit is not None:
            sql += " LIMIT ? OFFSET ?"
            params.extend([limit, max(0, offset)])
        elif offset:
            sql += " LIMIT -1 OFFSET ?"
            params.append(max(0, offset))
        with self._session() as c:
            rows = c.execute(sql, params).fetchall()
        docs = [dict(r) for r in rows]
        tags = self._tags_for([d["doc_id"] for d in docs])
        for d in docs:
            d["tags"] = tags.get(d["doc_id"], [])
        return docs

    def count_matching(self, doc_type: str | None = None,
                       source: str | None = None,
                       query: str | None = None) -> int:
        where, params = self._filter_sql(doc_type, source, query)
        with self._session() as c:
            row = c.execute(
                f"SELECT COUNT(*) FROM documents {where}".rstrip(),
                params).fetchone()
        return int(row[0])

    def count(self) -> int:
        with self._session() as c:
            return int(c.execute("SELECT COUNT(*) FROM documents").fetchone()[0])

    def delete(self, doc_id: str) -> bool:
        with self._session() as c:
            c.execute("DELETE FROM doc_tags WHERE doc_id=?", (doc_id,))
            c.execute("DELETE FROM watched_files WHERE doc_id=?", (doc_id,))
            cur = c.execute("DELETE FROM documents WHERE doc_id=?", (doc_id,))
        return cur.rowcount > 0

    def mark_missing(self, doc_id: str, missing: bool) -> None:
        with self._session() as c:
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
        with self._session() as c:
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
        out: dict[str, list[str]] = {}
        with self._session() as c:
            for batch in _batched(list(dict.fromkeys(doc_ids))):
                ph = ",".join("?" for _ in batch)
                rows = c.execute(
                    f"SELECT doc_id, tag FROM doc_tags WHERE doc_id IN ({ph}) "
                    "ORDER BY tag",
                    batch,
                ).fetchall()
                for did, tag in rows:
                    out.setdefault(did, []).append(tag)
        return out

    def all_tags(self) -> list[dict]:
        with self._session() as c:
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
        with self._session() as c:
            ph = ",".join("?" for _ in clean)
            rows = c.execute(
                f"""SELECT doc_id FROM doc_tags WHERE tag IN ({ph})
                    GROUP BY doc_id HAVING COUNT(DISTINCT tag)=?""",
                (*clean, len(clean)),
            ).fetchall()
        return [r[0] for r in rows]

    def doc_ids_for_source(self, source: str) -> list[str]:
        with self._session() as c:
            rows = c.execute(
                "SELECT doc_id FROM documents WHERE source=?", (source,)
            ).fetchall()
        return [r[0] for r in rows]

    def doc_ids_for_doc_type(self, doc_type: str) -> list[str]:
        dtype = normalize_doc_type(doc_type)
        with self._session() as c:
            rows = c.execute(
                "SELECT doc_id FROM documents WHERE doc_type=?", (dtype,)
            ).fetchall()
        return [r[0] for r in rows]

    def doc_ids_created_since(self, since: str) -> list[str]:
        """Doc ids created at/after an ISO date or datetime (recency scope).

        Raises ValueError when `since` is not parseable.
        """
        text = (since or "").strip()
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            raise ValueError(
                f"Bad 'since' value {since!r}: want an ISO date "
                "(e.g. 2026-01-15)") from None
        floor = parsed.isoformat()
        with self._session() as c:
            rows = c.execute(
                "SELECT doc_id FROM documents WHERE created_at >= ?",
                (floor,)).fetchall()
        return [r[0] for r in rows]

    # -- watched folders ------------------------------------------------
    def add_watch(self, path: str, recursive: bool = True) -> dict:
        now = datetime.now(timezone.utc).isoformat()
        with self._session() as c:
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
        with self._session() as c:
            rows = c.execute(
                "SELECT * FROM watch_folders ORDER BY created_at").fetchall()
        return [dict(r) for r in rows]

    def get_watch(self, watch_id: int) -> dict | None:
        with self._session() as c:
            row = c.execute(
                "SELECT * FROM watch_folders WHERE id=?", (watch_id,)).fetchone()
        return dict(row) if row else None

    def remove_watch(self, watch_id: int) -> bool:
        with self._session() as c:
            cur = c.execute(
                "DELETE FROM watch_folders WHERE id=?", (watch_id,))
        return cur.rowcount > 0

    def update_watch_status(self, watch_id: int, last_error: str = "") -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._session() as c:
            c.execute(
                "UPDATE watch_folders SET last_sync=?, last_error=? WHERE id=?",
                (now, last_error, watch_id),
            )

    def get_watched_file(self, path: str) -> dict | None:
        with self._session() as c:
            row = c.execute(
                "SELECT * FROM watched_files WHERE path=?", (path,)).fetchone()
        return dict(row) if row else None

    def set_watched_file(self, path: str, mtime: float, size: int,
                         content_hash: str, doc_id: str) -> None:
        with self._session() as c:
            c.execute(
                """INSERT INTO watched_files (path, mtime, size, content_hash, doc_id)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(path) DO UPDATE SET
                     mtime=excluded.mtime, size=excluded.size,
                     content_hash=excluded.content_hash, doc_id=excluded.doc_id""",
                (path, mtime, size, content_hash, doc_id),
            )

    def drop_watched_file(self, path: str) -> None:
        with self._session() as c:
            c.execute("DELETE FROM watched_files WHERE path=?", (path,))

    def watched_files_for(self, prefix: str) -> list[dict]:
        # Escape LIKE wildcards: '_' and '%' in folder names must match
        # literally, or /a/my_docs would also match /a/my-docs/...
        escaped = _escape_like(prefix.rstrip(os.sep))
        with self._session() as c:
            rows = c.execute(
                "SELECT * FROM watched_files WHERE path=? "
                "OR path LIKE ? ESCAPE '\\'",
                (prefix, escaped + os.sep + "%"),
            ).fetchall()
        return [dict(r) for r in rows]
