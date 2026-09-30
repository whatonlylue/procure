"""Folder watch: keep library documents in sync with directories on disk.

Poll-based (no watchdog dependency): every ``watch_interval`` seconds each
registered folder is scanned for new, changed, or removed files. State
(mtime/size/hash/doc_id per file) lives in the metadata store, so syncs
are incremental across restarts.

- New file -> ingested (tagged with nothing, ``source="watch"``).
- Changed file -> re-ingested into the same doc id (tags preserved).
- Removed file -> document marked ``missing`` (chunks stay searchable
  until the doc is deleted or the file returns).
"""
from __future__ import annotations

import hashlib
import os
import threading
from functools import lru_cache

from app import extract


def _iter_files(root: str, recursive: bool):
    if recursive:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            for name in filenames:
                if not name.startswith("."):
                    yield os.path.join(dirpath, name)
    else:
        try:
            names = os.listdir(root)
        except OSError:
            return
        for name in names:
            if name.startswith("."):
                continue
            full = os.path.join(root, name)
            if os.path.isfile(full):
                yield full


class WatchManager:
    def __init__(self, service=None):
        # service=None resolves lazily to get_service() (avoids import cycles
        # and lets tests inject an isolated RAGService).
        self._service = service
        self._lock = threading.Lock()
        self._stop: threading.Event | None = None
        self._thread: threading.Thread | None = None
        self.last_scan: str | None = None

    def _svc(self):
        if self._service is None:
            from app.service import get_service

            self._service = get_service()
        return self._service

    # -- folders ------------------------------------------------------
    def add(self, path: str, recursive: bool = True) -> dict:
        abspath = os.path.abspath(os.path.expanduser(path))
        if not os.path.isdir(abspath):
            raise ValueError(f"Not a directory: {path}")
        folder = self._svc().meta.add_watch(abspath, recursive)
        self.ensure_running()
        return folder

    def list(self) -> list[dict]:
        return self._svc().meta.list_watches()

    def remove(self, watch_id: int) -> bool:
        return self._svc().meta.remove_watch(watch_id)

    # -- sync ---------------------------------------------------------
    def sync_all(self, progress=None, cancelled=None) -> list[dict]:
        """Scan every watched folder once. Returns per-file result dicts."""
        from datetime import datetime, timezone

        svc = self._svc()
        folders = svc.meta.list_watches()
        results: list[dict] = []
        total = len(folders)
        for i, folder in enumerate(folders):
            if cancelled is not None and cancelled():
                break
            if progress is not None:
                progress(i, total, folder["path"])
            results.extend(self.sync_folder(folder["path"],
                                            bool(folder["recursive"]),
                                            svc=svc))
        if progress is not None:
            progress(total, total, "")
        self.last_scan = datetime.now(timezone.utc).isoformat()
        return results

    def sync_folder(self, root: str, recursive: bool = True, svc=None) -> list[dict]:
        svc = svc or self._svc()
        results: list[dict] = []
        seen: set[str] = set()
        for path in sorted(_iter_files(root, recursive)):
            if not extract.is_supported(path):
                continue
            seen.add(path)
            try:
                st = os.stat(path)
            except OSError:
                continue
            known = svc.meta.get_watched_file(path)
            if (known is not None and known["mtime"] == st.st_mtime
                    and known["size"] == st.st_size):
                # Unchanged; clear a stale "missing" flag if the file is back.
                if known["doc_id"]:
                    doc = svc.meta.get(known["doc_id"])
                    if doc is not None and doc["status"] == "missing":
                        svc.mark_missing(known["doc_id"], False)
                continue
            try:
                with open(path, "rb") as f:
                    data = f.read()
            except OSError as e:
                results.append({"path": path, "status": "failed",
                                "error": str(e)})
                continue
            digest = hashlib.sha256(data).hexdigest()
            if known is not None and known["content_hash"] == digest:
                svc.meta.set_watched_file(path, st.st_mtime, st.st_size,
                                          digest, known["doc_id"])
                if known["doc_id"]:
                    doc = svc.meta.get(known["doc_id"])
                    if doc is not None and doc["status"] == "missing":
                        svc.mark_missing(known["doc_id"], False)
                continue
            if known is not None and known["doc_id"]:
                res = svc.update_document_content(known["doc_id"], data)
                res["path"] = path
                svc.meta.set_watched_file(path, st.st_mtime, st.st_size,
                                          digest, known["doc_id"])
                svc.mark_missing(known["doc_id"], False)
                results.append(res)
            else:
                res = svc.ingest_path(path)
                res["path"] = path
                doc_id = res.get("doc_id", "")
                if res.get("status") in ("ready", "duplicate"):
                    svc.meta.set_watched_file(path, st.st_mtime, st.st_size,
                                              digest, doc_id)
                results.append(res)
        # Files tracked under this root that no longer exist -> missing.
        for row in svc.meta.watched_files_for(root):
            if row["path"] not in seen and os.path.isfile(row["path"]) is False:
                if row["doc_id"]:
                    doc = svc.meta.get(row["doc_id"])
                    if doc is not None and doc["status"] != "missing":
                        svc.mark_missing(row["doc_id"], True)
                        results.append({"path": row["path"], "status": "missing",
                                        "doc_id": row["doc_id"]})
        return results

    # -- background loop ----------------------------------------------
    def ensure_running(self) -> bool:
        """Start the poll loop when folders exist and interval > 0."""
        interval = self._svc().settings.watch_interval
        if interval <= 0 or not self._svc().meta.list_watches():
            return False
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return True
            self._stop = threading.Event()
            self._thread = threading.Thread(
                target=self._loop, args=(interval, self._stop),
                name="procure-watch", daemon=True)
            self._thread.start()
            return True

    def _loop(self, interval: int, stop: threading.Event) -> None:
        while not stop.wait(interval):
            try:
                self.sync_all()
            except Exception:  # noqa: BLE001 - watch loop must not die
                continue

    def stop(self) -> None:
        with self._lock:
            stop, thread = self._stop, self._thread
            self._stop, self._thread = None, None
        if stop is not None:
            stop.set()
        if thread is not None:
            thread.join(timeout=5)


@lru_cache(maxsize=1)
def get_watch_manager() -> WatchManager:
    return WatchManager()
