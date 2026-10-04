"""In-memory ring buffer for workspace server log lines.

Feeds GET /api/logs so the UI can show a workspace log view next to the
MCP server output. Best-effort and bounded; a restart clears it.
"""
from __future__ import annotations

import collections
import logging
import threading

_MAX_LINES = 1000


class LogBuffer(logging.Handler):
    def __init__(self, max_lines: int = _MAX_LINES):
        super().__init__()
        self._lock = threading.Lock()
        self._lines: collections.deque = collections.deque(maxlen=max_lines)
        self._seq = 0
        self.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            text = self.format(record)
        except Exception:
            return
        with self._lock:
            self._seq += 1
            self._lines.append((self._seq, text))

    def read(self, since: int = 0) -> dict:
        with self._lock:
            lines = [{"seq": s, "text": t} for s, t in self._lines if s > since]
            return {"lines": lines, "next": self._seq}


_BUFFER = LogBuffer()


def get_buffer() -> LogBuffer:
    return _BUFFER


def install() -> LogBuffer:
    """Attach the ring buffer to the root logger (idempotent)."""
    root = logging.getLogger()
    if _BUFFER not in root.handlers:
        root.addHandler(_BUFFER)
    return _BUFFER
