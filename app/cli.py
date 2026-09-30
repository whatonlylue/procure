"""procure desktop entry point: workspace server + MCP server in one binary.

Used two ways:

- From source: ``python -m app.cli serve [--port 8000] [--data-dir ...]``
- Frozen (PyInstaller onefile, managed by the Tauri shell): the sidecar
  binary IS this module, and the MCP server is a re-invocation of the same
  binary via the ``mcp-server`` subcommand.

``serve`` prints ``PROCURE_URL=http://host:port`` to stdout once the server
is bound (resolving ``--port 0`` to the real ephemeral port) so a launcher
can open the window on the right URL.
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import webbrowser

from app.version import __version__


def _parent_pid_alive(pid: int) -> bool:
    """True when a parent pid still exists (Windows orphan detection)."""
    if sys.platform == "win32":
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(0x1000, False, pid)
            if not handle:
                return False
            kernel32.CloseHandle(handle)
            return True
        except Exception:  # ctypes unavailable: assume alive, keep serving
            return True
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False
    except Exception:  # pragma: no cover - exotic platforms
        return True


def _watch_parent(server=None) -> None:
    """Exit when our parent dies instead of orphaning.

    A frozen onefile binary is a bootloader parent plus a payload child;
    whoever kills us (Tauri on quit, the MCP manager on stop) only reaches
    the bootloader, so the payload would be reparented and serve forever.
    On POSIX, reparenting shows up as getppid() == 1; on Windows getppid
    never becomes 1, so watch the recorded parent pid itself. When the
    parent is gone, the workspace server shuts down gracefully (running
    lifespan cleanup, which stops the MCP server); the MCP server spawns
    nothing, so it can exit immediately.
    """
    parent = os.getppid()
    if sys.platform != "win32" and parent == 1:
        return  # launched by init/launchd itself; nothing to watch

    def watch() -> None:
        while True:
            time.sleep(2.0)
            if sys.platform == "win32":
                alive = _parent_pid_alive(parent)
            else:
                alive = os.getppid() == parent or _parent_pid_alive(parent)
            if alive:
                continue
            if server is not None:
                server.should_exit = True
            else:
                os._exit(0)
            return

    threading.Thread(target=watch, name="parent-watch", daemon=True).start()


def resolve_data_dir(explicit: str | None) -> str:
    """One data-dir resolution for every subcommand.

    Explicit --data-dir wins, then PROCURE_DATA_DIR, then the OS
    user-data dir — so a harness-registered MCP server without env flags
    reads the same library as the desktop app instead of an empty ./data.
    """
    from app.config import _default_data_dir

    if explicit:
        chosen = explicit
    else:
        chosen = _default_data_dir()
    chosen = os.path.abspath(os.path.expanduser(chosen))
    os.environ["PROCURE_DATA_DIR"] = chosen
    os.makedirs(chosen, exist_ok=True)
    return chosen


def _parser() -> argparse.ArgumentParser:
    from app.config import get_settings

    defaults = get_settings()
    p = argparse.ArgumentParser(prog="procure", description="procure local RAG app")
    p.add_argument("--version", action="version", version=f"procure {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("serve", help="run the workspace web server")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000,
                   help="port to bind; 0 picks an ephemeral port")
    s.add_argument("--data-dir", default=None,
                   help="documents + databases (default: PROCURE_DATA_DIR or "
                        "the OS user-data dir)")
    s.add_argument("--open-browser", action="store_true",
                   help="open the workspace URL in a browser once bound")

    m = sub.add_parser("mcp-server", help="run the MCP server (Streamable HTTP)")
    m.add_argument("--host", default=defaults.mcp_host)
    m.add_argument("--port", type=int, default=defaults.mcp_port)
    m.add_argument("--data-dir", default=None,
                   help="same meaning as serve --data-dir")

    k = sub.add_parser("install-skills",
                       help="copy the procure agent skill into harness skill dirs")
    k.add_argument("--force", action="store_true",
                   help="overwrite existing installs (picks up skill updates)")
    k.add_argument("--targets", default=None,
                   help="comma-separated target ids (default: all)")

    sub.add_parser("skills-status",
                   help="show procure skill install state per harness")
    sub.add_parser("uninstall-skills",
                   help="remove the procure agent skill from harness skill dirs")

    a = sub.add_parser("add", help="ingest a file, URL, or stdin text")
    a.add_argument("source", help="file path, http(s) URL, or - for stdin")
    a.add_argument("--tags", default="",
                   help="comma-separated tags")
    a.add_argument("--doc-type", default="document",
                   help="document or memory")
    a.add_argument("--title", default=None,
                   help="title for stdin text (default: stdin)")
    a.add_argument("--data-dir", default=None,
                   help="same meaning as serve --data-dir")

    li = sub.add_parser("list", help="list stored documents")
    li.add_argument("--doc-type", default=None)
    li.add_argument("--source", default=None)
    li.add_argument("--query", default=None)
    li.add_argument("--limit", type=int, default=50)
    li.add_argument("--data-dir", default=None,
                    help="same meaning as serve --data-dir")

    se = sub.add_parser("search", help="search the library")
    se.add_argument("query")
    se.add_argument("--top-k", type=int, default=5)
    se.add_argument("--doc-type", default=None)
    se.add_argument("--tags", default="",
                    help="comma-separated tags (AND semantics)")
    se.add_argument("--data-dir", default=None,
                    help="same meaning as serve --data-dir")
    return p


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from app.main import app as fastapi_app

    data_dir = resolve_data_dir(args.data_dir)

    config = uvicorn.Config(fastapi_app, host=args.host, port=args.port,
                            log_level="info")
    server = uvicorn.Server(config)
    _watch_parent(server)
    orig_startup = server.startup

    async def startup_and_announce(sockets=None):
        await orig_startup(sockets)
        bound = args.port
        try:
            bound = server.servers[0].sockets[0].getsockname()[1]
        except Exception:  # fall back to configured port
            pass
        url = f"http://{args.host}:{bound}"
        print(f"PROCURE_URL={url}", flush=True)
        print(f"procure {__version__} serving {data_dir}", flush=True)
        if args.open_browser:
            webbrowser.open(url)

    server.startup = startup_and_announce  # type: ignore[method-assign]
    server.run()
    return 0


def cmd_mcp_server(args: argparse.Namespace) -> int:
    from app.mcp_server import main as mcp_main

    # Always resolve: a harness-registered MCP server without env flags
    # must read the desktop library, not an empty ./data.
    resolve_data_dir(args.data_dir)
    _watch_parent()
    mcp_main(["--host", args.host, "--port", str(args.port)])
    return 0


def _service_for(args: argparse.Namespace):
    from app.config import get_settings
    from app.service import RAGService

    resolve_data_dir(getattr(args, "data_dir", None))
    return RAGService(get_settings())


def cmd_install_skills(args: argparse.Namespace) -> int:
    from app import skills

    target_ids = ([t for t in (args.targets or "").split(",") if t.strip()]
                  or None)
    try:
        out = skills.install(target_ids, args.force)
    except (ValueError, FileNotFoundError) as e:
        print(f"error: {e}")
        return 1
    failed = 0
    for r in out["results"]:
        if r["action"] == "error":
            failed += 1
            print(f"{r['id']}: ERROR {r['detail']} ({r['path']})")
        elif r["action"] == "skipped":
            print(f"{r['id']}: skipped ({r['detail']}) — {r['path']}")
        else:
            print(f"{r['id']}: {r['action']} — {r['path']}")
    return 1 if failed else 0


def cmd_skills_status(args: argparse.Namespace) -> int:
    from app import skills

    del args
    st = skills.status()
    if not st["source_found"]:
        print("bundled skill: NOT FOUND")
    else:
        print(f"bundled skill: {st['skill']} v{st['version'] or '?'}")
    for t in st["targets"]:
        state = "not installed"
        if t["installed"]:
            state = f"installed v{t['installed_version'] or '?'}"
            if t["outdated"]:
                state += " (update available)"
        print(f"  {t['id']} [{t['label']}]: {state} — {t['path']}")
    return 0


def cmd_uninstall_skills(args: argparse.Namespace) -> int:
    from app import skills

    del args
    out = skills.uninstall()
    failed = 0
    for r in out["results"]:
        if r["action"] == "error":
            failed += 1
            print(f"{r['id']}: ERROR {r['detail']} ({r['path']})")
        else:
            print(f"{r['id']}: {r['action']} — {r['path']}")
    return 1 if failed else 0


def cmd_add(args: argparse.Namespace) -> int:
    from app.store import normalize_doc_type

    try:
        dtype = normalize_doc_type(args.doc_type)
    except ValueError as e:
        print(f"error: {e}")
        return 1
    tags = [t.strip() for t in (args.tags or "").split(",") if t.strip()]
    svc = _service_for(args)
    if args.source.startswith(("http://", "https://")):
        res = svc.ingest_url(args.source, tags, dtype)
    elif args.source == "-":
        text = sys.stdin.read()
        if not text.strip():
            print("error: no text on stdin")
            return 1
        res = svc.ingest_text(args.title or "stdin", text, tags, dtype)
    else:
        if not os.path.isfile(args.source):
            print(f"error: not a file: {args.source}")
            return 1
        with open(args.source, "rb") as f:
            data = f.read()
        res = svc.ingest_files([(os.path.basename(args.source), data)],
                               tags=tags, doc_type=dtype)[0]
    if res.get("status") == "failed":
        print(f"failed: {res.get('error')}")
        return 1
    print(f"{res.get('status')}: {res.get('doc_id')} "
          f"{res.get('filename')} ({res.get('chunk_count', 0)} chunks)")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    svc = _service_for(args)
    try:
        docs = svc.list_documents(args.doc_type, args.source, args.query,
                                  args.limit, 0)
    except ValueError as e:
        print(f"error: {e}")
        return 1
    for d in docs:
        print(f"{d['doc_id']} [{d['status']}] {d['chunk_count']:>4} chunks "
              f"{d.get('doc_type', 'document'):>8} {d['filename']}")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    svc = _service_for(args)
    tags = [t.strip() for t in (args.tags or "").split(",") if t.strip()] or None
    try:
        hits = svc.search(args.query, args.top_k, None, tags,
                          None, args.doc_type)
    except ValueError as e:
        print(f"error: {e}")
        return 1
    for i, h in enumerate(hits, 1):
        print(f"#{i} [{h['score']:.3f}|ans {h['answerability']:.2f}] "
              f"{h['filename']} ({h['chunk_id']})")
        print(f"    {h['text'][:280].replace(chr(10), ' ')}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "serve":
        return cmd_serve(args)
    if args.command == "mcp-server":
        return cmd_mcp_server(args)
    if args.command == "install-skills":
        return cmd_install_skills(args)
    if args.command == "skills-status":
        return cmd_skills_status(args)
    if args.command == "uninstall-skills":
        return cmd_uninstall_skills(args)
    if args.command == "add":
        return cmd_add(args)
    if args.command == "list":
        return cmd_list(args)
    if args.command == "search":
        return cmd_search(args)
    raise AssertionError(f"unhandled command {args.command}")


if __name__ == "__main__":
    sys.exit(main())
