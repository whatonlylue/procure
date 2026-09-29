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


def _watch_parent(server=None) -> None:  # noqa: ANN001
    """Exit when our parent dies instead of orphaning.

    A frozen onefile binary is a bootloader parent plus a payload child;
    whoever kills us (Tauri on quit, the MCP manager on stop) only reaches
    the bootloader, so the payload would be reparented to init and serve
    forever. When reparenting is detected, the workspace server shuts down
    gracefully (running lifespan cleanup, which stops the MCP server); the
    MCP server spawns nothing, so it can exit immediately.
    """
    if os.getppid() == 1:
        return  # launched by init/launchd itself; nothing to watch

    def watch() -> None:
        while True:
            time.sleep(2.0)
            if os.getppid() != 1:
                continue
            if server is not None:
                server.should_exit = True
            else:
                os._exit(0)
            return

    threading.Thread(target=watch, name="parent-watch", daemon=True).start()


def default_data_dir() -> str:
    from platformdirs import user_data_dir

    return user_data_dir("procure")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="procure", description="procure local RAG app")
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
    m.add_argument("--host", default=os.environ.get("PROCURE_MCP_HOST", "127.0.0.1"))
    m.add_argument("--port", type=int,
                   default=int(os.environ.get("PROCURE_MCP_PORT", "8001")))
    m.add_argument("--data-dir", default=None,
                   help="same meaning as serve --data-dir")
    return p


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from app.main import app as fastapi_app

    data_dir = os.path.abspath(os.path.expanduser(
        args.data_dir or os.environ.get("PROCURE_DATA_DIR") or default_data_dir()))
    # Export for this process (settings) and children (MCP subprocess).
    os.environ["PROCURE_DATA_DIR"] = data_dir
    os.makedirs(data_dir, exist_ok=True)

    config = uvicorn.Config(fastapi_app, host=args.host, port=args.port,
                            log_level="info")
    server = uvicorn.Server(config)
    _watch_parent(server)
    orig_startup = server.startup

    async def startup_and_announce(sockets=None):  # noqa: ANN001, ANN202
        await orig_startup(sockets)
        bound = args.port
        try:
            bound = server.servers[0].sockets[0].getsockname()[1]
        except Exception:  # noqa: BLE001 - fall back to configured port
            pass
        url = f"http://{args.host}:{bound}"
        print(f"PROCURE_URL={url}", flush=True)
        if args.open_browser:
            webbrowser.open(url)

    server.startup = startup_and_announce  # type: ignore[method-assign]
    server.run()
    return 0


def cmd_mcp_server(args: argparse.Namespace) -> int:
    from app.mcp_server import main as mcp_main

    if args.data_dir or os.environ.get("PROCURE_DATA_DIR"):
        data_dir = os.path.abspath(os.path.expanduser(
            args.data_dir or os.environ["PROCURE_DATA_DIR"]))
        os.environ["PROCURE_DATA_DIR"] = data_dir
        os.makedirs(data_dir, exist_ok=True)
    _watch_parent()
    mcp_main(["--host", args.host, "--port", str(args.port)])
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "serve":
        return cmd_serve(args)
    if args.command == "mcp-server":
        return cmd_mcp_server(args)
    raise AssertionError(f"unhandled command {args.command}")  # noqa: TRY003


if __name__ == "__main__":
    sys.exit(main())
