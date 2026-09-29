"""SQLite metadata store for documents."""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone


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

    def _connect(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.path)
        c.row_factory = sqlite3.Row
        return c

    def upsert(self, doc_id: str, filename: str, status: str,
               chunk_count: int = 0, error: str = "") -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as c:
            c.execute(
                """INSERT INTO documents (doc_id, filename, status, chunk_count, error, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(doc_id) DO UPDATE SET
                     filename=excluded.filename, status=excluded.status,
                     chunk_count=excluded.chunk_count, error=excluded.error""",
                (doc_id, filename, status, chunk_count, error, now),
            )

    def get(self, doc_id: str) -> dict | None:
        with self._connect() as c:
            row = c.execute("SELECT * FROM documents WHERE doc_id=?", (doc_id,)).fetchone()
        return dict(row) if row else None

    def list(self) -> list[dict]:
        with self._connect() as c:
            rows = c.execute("SELECT * FROM documents ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]

    def delete(self, doc_id: str) -> bool:
        with self._connect() as c:
            cur = c.execute("DELETE FROM documents WHERE doc_id=?", (doc_id,))
        return cur.rowcount > 0
