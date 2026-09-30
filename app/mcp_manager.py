"""Lifecycle manager for the procure MCP server subprocess.

The MCP server runs as a child process (`python -m app.mcp_server`) serving
Streamable HTTP. stdout/stderr are merged and kept in a ring buffer so the
workspace UI can show a live server-output window.
"""
from __future__ import annotations

import atexit
import collections
import os
import subprocess
import sys
import threading
import time
from functools import lru_cache

from app.config import get_settings

_MAX_LINES = 1000


def _project_root() -> str:
    """Repo root for a source checkout (parent of app/)."""
    from pathlib import Path

    return str(Path(__file__).resolve().parent.parent)


def _server_cmd(settings) -> list[str]:
    """Command line that (re)starts the MCP server in this environment."""
    if getattr(sys, "frozen", False):
        # PyInstaller onefile: re-invoke this binary's mcp-server subcommand.
        return [sys.executable, "mcp-server",
                "--host", settings.mcp_host, "--port", str(settings.mcp_port)]
    return [sys.executable, "-u", "-m", "app.mcp_server",
            "--host", settings.mcp_host, "--port", str(settings.mcp_port)]


def _server_env() -> dict:
    """Environment for the MCP subprocess.

    cwd is the data dir (absolute, possibly read-only when frozen), so a
    plain source checkout needs PYTHONPATH pointed at the project root or
    `python -m app.mcp_server` fails with "No module named app".
    """
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    if not getattr(sys, "frozen", False):
        root = _project_root()
        prev = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = root + (os.pathsep + prev if prev else "")
    return env


class MCPManager:
    def __init__(self, max_lines: int = _MAX_LINES):
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._started_at: float | None = None
        self._exit_code: int | None = None
        self._lines: collections.deque = collections.deque(maxlen=max_lines)
        self._seq = 0
        atexit.register(self._shutdown)

    # -- public API ----------------------------------------------------
    def status(self) -> dict:
        with self._lock:
            self._reap_locked()
            settings = get_settings()
            return {
                "running": self._proc is not None,
                "pid": self._proc.pid if self._proc else None,
                "url": f"http://{settings.mcp_host}:{settings.mcp_port}/mcp",
                "uptime_s": round(time.time() - self._started_at, 1)
                if self._proc and self._started_at else 0.0,
                "exit_code": self._exit_code,
            }

    def start(self) -> dict:
        with self._lock:
            self._reap_locked()
            if self._proc is not None:
                return self.status()
            settings = get_settings()
            env = _server_env()
            # Carry the workspace's current retrieval settings into the
            # subprocess so MCP search matches the UI. (Restart the server
            # after changing them; local import avoids a service cycle.
            # Persisted settings.json is the primary channel; env covers
            # unsaved runtime tweaks.)
            try:
                from app.service import get_service

                svc_settings = get_service().settings
                env["PROCURE_SEARCH"] = svc_settings.search_mode
                env["PROCURE_RERANK"] = svc_settings.rerank
            except Exception:  # fall back to ambient env
                pass
            self._exit_code = None
            os.makedirs(settings.data_dir, exist_ok=True)
            self._proc = subprocess.Popen(
                _server_cmd(settings),
                # data_dir is absolute; the bundle dir is read-only when frozen.
                cwd=settings.data_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            self._started_at = time.time()
            threading.Thread(target=self._drain, daemon=True).start()
            pid = self._proc.pid
        self._append(f"[manager] started MCP server (pid {pid})")
        return self.status()

    def stop(self) -> dict:
        with self._lock:
            proc = self._proc
            self._proc = None
            self._started_at = None
        if proc is None:
            return self.status()
        self._append(f"[manager] stopping MCP server (pid {proc.pid})")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
        with self._lock:
            self._exit_code = proc.returncode
        self._append(f"[manager] MCP server exited (code {proc.returncode})")
        return self.status()

    def logs(self, since: int = 0) -> dict:
        with self._lock:
            lines = [{"seq": s, "text": t} for s, t in self._lines if s > since]
            return {"lines": lines, "next": self._seq}

    # -- internals -----------------------------------------------------
    def _append(self, text: str) -> None:
        with self._lock:
            self._seq += 1
            self._lines.append((self._seq, text))

    def _drain(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        try:
            for line in proc.stdout:
                self._append(line.rstrip("\n"))
        except ValueError:
            pass  # stdout closed during shutdown
        finally:
            with self._lock:
                self._reap_locked()

    def _reap_locked(self) -> None:
        if self._proc is not None and self._proc.poll() is not None:
            self._exit_code = self._proc.returncode
            self._seq += 1
            self._lines.append(
                (self._seq,
                 f"[manager] MCP server exited unexpectedly (code {self._exit_code})"))
            self._proc = None
            self._started_at = None

    def _shutdown(self) -> None:
        try:
            self.stop()
        except Exception:
            pass


@lru_cache(maxsize=1)
def get_mcp_manager() -> MCPManager:
    return MCPManager()
