"""Background jobs: uploads, folder syncs, and URL/reingest work run here.

Jobs are in-memory (a server restart drops their history, never the
library itself). Cancellation is cooperative: queued jobs cancel
immediately, running jobs stop at the next file boundary.
"""
from __future__ import annotations

import concurrent.futures
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache

_MAX_JOBS = 200


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class JobHandle:
    """Passed to the job function for progress + cooperative cancellation."""

    job_id: str
    _manager: JobManager = field(repr=False)
    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)

    def update(self, done: int, total: int, current: str = "") -> None:
        self._manager._update(self.job_id, done, total, current)

    def is_cancelled(self) -> bool:
        return self._cancel.is_set()


class JobManager:
    def __init__(self, max_workers: int = 2):
        self._lock = threading.Lock()
        self._jobs: dict[str, dict] = {}
        self._order: list[str] = []
        self._handles: dict[str, JobHandle] = {}
        self._futures: dict[str, concurrent.futures.Future] = {}
        self._pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="procure-job")

    def submit(self, kind: str, label: str, func, *args, **kwargs) -> dict:
        """Run func(handle, *args, **kwargs) in the background; returns the job."""
        job_id = uuid.uuid4().hex[:12]
        handle = JobHandle(job_id, self)
        job = {
            "job_id": job_id, "kind": kind, "label": label,
            "status": "queued", "done": 0, "total": 0, "current": "",
            "result": None, "error": "",
            "created_at": _now(), "updated_at": _now(),
        }
        with self._lock:
            self._jobs[job_id] = job
            self._handles[job_id] = handle
            self._order.append(job_id)
            self._prune_locked()
            fut = self._pool.submit(self._run, job_id, func, args, kwargs)
            self._futures[job_id] = fut
        return dict(job)

    def _prune_locked(self) -> None:
        """Drop oldest finished jobs past the cap (active jobs stay put)."""
        while len(self._order) > _MAX_JOBS:
            victim = next(
                (jid for jid in self._order
                 if self._jobs.get(jid, {}).get("status")
                 in ("done", "failed", "cancelled")),
                None,
            )
            if victim is None:
                break  # all active: allow overflow rather than scramble order
            self._order.remove(victim)
            self._jobs.pop(victim, None)
            self._handles.pop(victim, None)
            self._futures.pop(victim, None)

    def _run(self, job_id: str, func, args, kwargs) -> None:
        with self._lock:
            handle = self._handles.get(job_id)
            job = self._jobs.get(job_id)
            if job is None:
                return
            if handle is not None and handle.is_cancelled():
                job.update(status="cancelled", updated_at=_now())
                return
            job.update(status="running", updated_at=_now())
        try:
            result = func(handle, *args, **kwargs)
        except Exception as e:  # job failure is data
            with self._lock:
                job = self._jobs.get(job_id)
                if job is not None:
                    cancelled = handle is not None and handle.is_cancelled()
                    job.update(
                        status="cancelled" if cancelled else "failed",
                        error="" if cancelled else str(e),
                        updated_at=_now(),
                    )
            return
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                if handle is not None and handle.is_cancelled():
                    job.update(status="cancelled", updated_at=_now())
                else:
                    job.update(status="done", result=result, updated_at=_now())

    def _update(self, job_id: str, done: int, total: int, current: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                job.update(done=done, total=total, current=current,
                           updated_at=_now())

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None

    def list(self) -> list[dict]:
        with self._lock:
            return [dict(self._jobs[j]) for j in reversed(self._order)
                    if j in self._jobs]

    def cancel(self, job_id: str) -> dict | None:
        """Cancel a job; returns the job, or None when unknown."""
        with self._lock:
            job = self._jobs.get(job_id)
            handle = self._handles.get(job_id)
            fut = self._futures.get(job_id)
            if job is None:
                return None
            if job["status"] in ("done", "failed", "cancelled"):
                return dict(job)
            if handle is not None:
                handle._cancel.set()
            cancelled_now = bool(fut is not None and fut.cancel())
            if cancelled_now:
                job.update(status="cancelled", updated_at=_now())
            return dict(job)

    def wait(self, job_id: str, timeout: float = 60.0) -> dict | None:
        """Block until a job finishes (tests + scripts)."""
        with self._lock:
            fut = self._futures.get(job_id)
        if fut is None:
            return self.get(job_id)
        try:
            fut.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            pass
        return self.get(job_id)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


@lru_cache(maxsize=1)
def get_job_manager() -> JobManager:
    return JobManager()
