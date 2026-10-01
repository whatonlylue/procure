#!/usr/bin/env python3
"""Propagate app/version.py to every file that carries the version string.

Usage (from the repo root, after bumping ``app/version.py``)::

    python scripts/sync_version.py

Files updated: pyproject.toml, frontend/package.json,
src-tauri/Cargo.toml, src-tauri/tauri.conf.json,
.claude-plugin/plugin.json, .claude-plugin/marketplace.json.
The frontend UI, fetch user agent, and ``--version`` read the version
at runtime from the backend, so they need no sync.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.version import __version__  # noqa: E402


def _sub(path: Path, pattern: str, repl: str) -> None:
    text = path.read_text(encoding="utf-8")
    new, n = re.subn(pattern, repl, text, count=1)
    if n != 1:
        raise SystemExit(f"pattern not found in {path}: {pattern}")
    path.write_text(new, encoding="utf-8")
    print(f"  {path.relative_to(ROOT)}")


def main() -> None:
    v = __version__
    print(f"syncing version {v} ...")
    _sub(ROOT / "pyproject.toml", r'(?m)^version = "[^"]+"$',
         f'version = "{v}"')
    _sub(ROOT / "frontend" / "package.json", r'"version": "[^"]+"',
         f'"version": "{v}"')
    _sub(ROOT / "src-tauri" / "Cargo.toml", r'(?m)^version = "[^"]+"$',
         f'version = "{v}"')

    _sub(ROOT / "src-tauri" / "tauri.conf.json",
         r'"version": "[^"]+"', f'"version": "{v}"')

    for name in ("plugin.json", "marketplace.json"):
        p = ROOT / ".claude-plugin" / name
        data = json.loads(p.read_text(encoding="utf-8"))
        if name == "plugin.json":
            data["version"] = v
        else:
            for entry in data.get("plugins", []):
                entry["version"] = v
        p.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        print(f"  {p.relative_to(ROOT)}")
    print("done.")


if __name__ == "__main__":
    main()
