"""Folder watch: keep library documents in sync with directories on disk.

Poll-based (no watchdog dependency): every ``watch_interval`` seconds each
registered folder is scanned for new, changed, or removed files. State
(mtime/size/hash/doc_id per file) lives in the metadata store, so syncs
are incremental across restarts.

- New file -> ingested (``source="watch"``, ``source_uri`` = the path).
- Changed file -> re-ingested into the same doc id (tags preserved).
- Removed file -> document marked ``missing`` (chunks stay searchable
  until the doc is deleted or the file returns).

A watched path maps to the doc it owns (that doc's ``source_uri`` is the
path). When a file's content duplicates some other document, the path is
still recorded (so polls stay quiet) but the foreign doc is never
overwritten or marked missing on this path's behalf.
"""
from __future__ import annotations

import hashlib
import logging
import os
import threading
from datetime import datetime, timezone
from functools import lru_cache

from app import extract

logger = logging.getLogger(__name__)


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
        self._sync_lock = threading.Lock()
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
        folders = self._svc().meta.list_watches()
        for folder in folders:
            rows = self._svc().meta.watched_files_for(folder["path"])
            folder["file_count"] = len(rows)
        return folders

    def remove(self, watch_id: int, delete_docs: bool = False) -> bool:
        svc = self._svc()
        folder = svc.meta.get_watch(watch_id)
        if folder is None:
            return False
        if delete_docs:
            for row in svc.meta.watched_files_for(folder["path"]):
                doc_id = row.get("doc_id") or ""
                if not doc_id:
                    continue
                doc = svc.meta.get(doc_id)
                # Only delete docs this path owns; a linked duplicate may
                # belong to another file (or an upload/memory).
                if doc is not None and doc.get("source_uri") == row["path"]:
                    svc.delete_document(doc_id)
                svc.meta.drop_watched_file(row["path"])
        return svc.meta.remove_watch(watch_id)

    # -- sync ---------------------------------------------------------
    def sync_all(self, progress=None, cancelled=None) -> list[dict]:
        """Scan every watched folder once. Returns per-file result dicts."""
        # The background loop and an explicit "Sync now" job must not scan
        # concurrently: both could ingest the same new file before either
        # commits its content hash, minting duplicate docs.
        with self._sync_lock:
            return self._sync_all_locked(progress, cancelled)

    def _sync_all_locked(self, progress=None, cancelled=None) -> list[dict]:
        svc = self._svc()
        folders = svc.meta.list_watches()
        results: list[dict] = []
        total = len(folders)
        for i, folder in enumerate(folders):
            if cancelled is not None and cancelled():
                break
            if progress is not None:
                progress(i, total, folder["path"])
            try:
                folder_results = self.sync_folder(folder["path"],
                                                  bool(folder["recursive"]),
                                                  svc=svc)
            except Exception as e:  # one folder must not abort the rest
                logger.warning("watch sync failed for %s: %s",
                               folder["path"], e)
                svc.meta.update_watch_status(folder["id"], str(e))
                continue
            errors = [r.get("error", "") for r in folder_results
                      if r.get("status") == "failed" and r.get("error")]
            svc.meta.update_watch_status(folder["id"], errors[0] if errors else "")
            results.extend(folder_results)
        if progress is not None:
            progress(total, total, "")
        self.last_scan = datetime.now(timezone.utc).isoformat()
        return results

    def _clear_stale_missing(self, svc, doc_id: str) -> None:
        """Clear a "missing" flag when the file is back (shared helper)."""
        if not doc_id:
            return
        doc = svc.meta.get(doc_id)
        if doc is not None and doc["status"] == "missing":
            svc.mark_missing(doc_id, False)

    def sync_folder(self, root: str, recursive: bool = True, svc=None) -> list[dict]:
        svc = svc or self._svc()
        results: list[dict] = []
        seen: set[str] = set()
        # One query per folder (not per file) for known state.
        known_map = {r["path"]: r for r in svc.meta.watched_files_for(root)}
        for path in sorted(_iter_files(root, recursive)):
            if not extract.is_supported(path):
                continue
            seen.add(path)
            try:
                results.extend(self._sync_one(svc, path, known_map.get(path)))
            except Exception as e:  # one file must not abort the folder
                logger.warning("watch sync failed for %s: %s", path, e)
                results.append({"path": path, "status": "failed",
                                "error": str(e)})
        # Files tracked under this root that no longer exist -> missing.
        for row in svc.meta.watched_files_for(root):
            if row["path"] in seen or os.path.isfile(row["path"]):
                continue
            if not row["doc_id"]:
                continue
            doc = svc.meta.get(row["doc_id"])
            if doc is None or doc["status"] == "missing":
                continue
            # Only the owning path may mark a doc missing: a linked
            # duplicate (same content as another file) must not flag a
            # doc whose own file still exists.
            if doc.get("source_uri") != row["path"]:
                continue
            svc.mark_missing(row["doc_id"], True)
            results.append({"path": row["path"], "status": "missing",
                            "doc_id": row["doc_id"]})
        return results

    def _sync_one(self, svc, path: str, known: dict | None) -> list[dict]:
        try:
            st = os.stat(path)
        except OSError:
            return []
        if (known is not None and known["mtime"] == st.st_mtime
                and known["size"] == st.st_size):
            # Unchanged; clear a stale "missing" flag if the file is back.
            self._clear_stale_missing(svc, known["doc_id"])
            return []
        try:
            with open(path, "rb") as f:
                data = f.read()
        except OSError as e:
            return [{"path": path, "status": "failed", "error": str(e)}]
        digest = hashlib.sha256(data).hexdigest()
        if known is not None and known["content_hash"] == digest:
            svc.meta.set_watched_file(path, st.st_mtime, st.st_size,
                                      digest, known["doc_id"])
            self._clear_stale_missing(svc, known["doc_id"])
            return []
        if known is not None and known["doc_id"]:
            doc = svc.meta.get(known["doc_id"])
            if doc is None:
                # The doc was deleted from the Library but the row survived
                # (older DBs): drop the stale row and ingest as new.
                svc.meta.drop_watched_file(path)
                return self._ingest_new(svc, path, st, digest)
            if doc.get("source_uri") != path:
                # Linked duplicate: this path never owned the mapped doc
                # (an upload, a memory, or another file's content), so a
                # change here must ingest fresh, never overwrite it.
                return self._ingest_new(svc, path, st, digest)
            try:
                res = svc.update_document_content(known["doc_id"], data)
            except KeyError:
                svc.meta.drop_watched_file(path)
                return self._ingest_new(svc, path, st, digest)
            res["path"] = path
            svc.meta.set_watched_file(path, st.st_mtime, st.st_size,
                                      digest, known["doc_id"])
            svc.mark_missing(known["doc_id"], False)
            return [res]
        return self._ingest_new(svc, path, st, digest)

    def _ingest_new(self, svc, path: str, st, digest: str) -> list[dict]:
        res = svc.ingest_path(path)
        res["path"] = path
        doc_id = res.get("doc_id", "")
        # Record every outcome with a doc id -- including failures. Only
        # "ready"/"duplicate" used to be recorded, so an unsupported-by-OCR
        # file (images in the desktop build, which excludes OCR) minted a
        # fresh failed doc on EVERY poll, growing the library without bound.
        if doc_id:
            svc.meta.set_watched_file(path, st.st_mtime, st.st_size,
                                      digest, doc_id)
        return [res]

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
                target=self._loop, args=(self._stop,),
                name="procure-watch", daemon=True)
            self._thread.start()
            return True

    def _loop(self, stop: threading.Event) -> None:
        # The interval is re-read every iteration so a settings change
        # takes effect without a restart.
        while True:
            try:
                interval = max(1, self._svc().settings.watch_interval)
            except Exception:  # settings failure must not kill the loop
                interval = 60
            if stop.wait(interval):
                return
            try:
                self.sync_all()
            except Exception:  # watch loop must not die
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
