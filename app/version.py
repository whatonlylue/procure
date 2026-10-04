"""Single source of truth for the procure version.

Release bumps edit ONLY this file, then run ``scripts/sync_version.py``
to propagate to pyproject.toml, frontend/package.json,
src-tauri/Cargo.toml, src-tauri/tauri.conf.json, and the plugin
manifests. Runtime code (``/api/health``, the fetch user agent,
``--version``) imports from here so it can never drift.
"""

__version__ = "0.2.4"
