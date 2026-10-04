# procure packaging research

Goal: a download-and-run procure for non-expert users, macOS first,
Windows/Linux nice-to-have. Researched Sep 2026 (tool states as of 2025–26).

**Status: Tauri shell built and verified locally (Apple Silicon).**
`src-tauri/target/release/bundle/macos/Procure.app` (40MB) and
`.../bundle/dmg/Procure_0.1.0_aarch64.dmg` (32.6MB) are produced by
`./scripts/build_desktop.sh` (frontend → PyInstaller onefile sidecar,
29MB, torch excluded → Tauri bundle). Verified: launch, ephemeral-port
serve, ingest + search + full-doc fetch through the app, data persisted
in `~/Library/Application Support/com.procure.app/`, clean quit with no
orphaned sidecar (parent-watchdog in `app/cli.py` covers the onefile
bootloader/payload split). NOTE: since embeddings moved to ONNX Runtime
granite (app/embeddings.py), the sidecar must now bundle `onnxruntime` +
`tokenizers` + `huggingface_hub` (see procure-sidecar.spec) and the
numbers above are stale — re-verify size and the first-run model download
before the next release. v0.1 ships unsigned macOS (`.dmg`) + Windows
(NSIS `.exe`) bundles, built by `.github/workflows/release.yml` on every
`v*` tag push. Remaining before a public release: signing +
notarization (needs a $99 Apple Developer ID), updater plugin, Intel +
Linux builds, final icon (current one is a placeholder).

## App constraints driving the choice

- FastAPI + built Vite statics; `app/mcp_manager.py` spawns
  `[sys.executable, "-m", "app.mcp_server"]` with `cwd=<project root>` —
  any frozen build must preserve "same interpreter, same env, known cwd"
  (frozen exe re-invoked via `sys.executable` + a `--mcp-server`-style
  subcommand; needs a small code change).
- Data dir (`PROCURE_DATA_DIR`, default `data/`) must move to a
  user-writable location outside the bundle (`platformdirs.user_data_dir`,
  e.g. `~/Library/Application Support/procure`) or upgrades wipe user data.
  Mandatory before any packaged release.
- Heavy optionals (`sentence-transformers` → torch ~2GB, for the optional
  neural reranker) must stay opt-in; never in the base artifact. The
  cached-only model-download defaults are already exactly right for
  distribution.

## Options compared

| Option | Build story | Size | macOS signing | Auto-update | Verdict |
|---|---|---|---|---|---|
| **uv one-liner + PyPI** (`uv tool install procure`) | Publish wheel incl. prebuilt frontend; install script bootstraps uv, installs, opens browser | ~100MB deps on install | None (no binary) | Manual (`uv tool upgrade`) | **Do first** — zero signing/CI pain, all OSes, unblocks trying it this week |
| **Tauri 2.x + PyInstaller sidecar** | Proven 2025–26 pattern for FastAPI apps; `beforeBuildCommand` builds frontend + sidecar; Tauri manages sidecar lifecycle | ~100–150MB | Medium (documented CI flow; needs $99 Apple Developer ID) | Built-in updater plugin (best story) | **The real artifact** — native window + updater; macOS Apple Silicon + Intel first |
| **Docker image** | Trivial multi-stage build once PyPI packaging exists | ~500MB–1GB | N/A | `docker pull` | Cheap power-user alternative; nearly free after PyPI |
| **Electron + bundled Python** | Same sidecar approach, heavier shell | ~200–300MB+ | Medium, well-documented | Excellent | Rejected — Tauri does the same smaller |
| **pyapp** (active, v0.29 Oct 2025) | 3MB Rust launcher bootstrapping CPython via uv at first run | 3MB + ~300MB bootstrap | Low | None | Rejected — first-run download violates offline-first; needs PyPI anyway |
| **Briefcase** (active, 3.14 support) | Native-UI-oriented; server+localhost-browser fights its model | ~100–200MB | Low-medium (built-in support) | None | Rejected — wrong UI model |
| **conda constructor** | Wizard-style installer with full env | 300MB+ | Medium | None | Fallback only; dated UX |

Also checked: PyOxidizer is unmaintained (skip); PyInstaller works with
`python-build-standalone`/uv Pythons where py2app fails.

## Plan

Note: we skipped straight to Step 3 (Tauri) — PyPI/Docker (Steps 1–2)
are deferred until/unless a terminal-based install path is wanted. The
`app/cli.py` entry point + `platformdirs` data dir built for the sidecar
already cover most of Step 1's code changes.

**Step 1 — pip-installable with bundled frontend (1–2 days).**
Add `[project.scripts] procure = "app.cli:main"` + small `app/cli.py`
(start uvicorn on 127.0.0.1, open browser, `--port/--data-dir/--no-browser`).
Ship `app/static/` in the wheel (`package-data` already configured);
serve from `importlib.resources`, never relative paths. Default data dir
via `platformdirs` with `./data` migration. Add `--mcp-server` CLI
subcommand for frozen re-invocation. Publish to PyPI + `install.sh` /
`install.ps1` one-liners.

**Step 2 — Docker (half day).** Multi-stage Dockerfile (node build →
`python:3.12-slim` + wheel), `VOLUME /data`, GHCR push via CI.

**Step 3 — Tauri shell (~1 week, once users exist).** `cargo tauri init`
against `frontend/`; PyInstaller onefile sidecar of `app/cli.py`
(onnxruntime bundled for embeddings; torch still excluded);
`externalBin` + spawn on start with `--port 0`, kill on exit;
GitHub Actions matrix (macOS arm64/Intel, Windows, Linux) → signed
`.dmg`/`.msi`/`.AppImage` + updater endpoint.

**Defer:** signing/notarization (only needed at step 3), torch bundling
(keep opt-in), Windows/Linux native shells (macOS Tauri first, others via
uv/Docker).

## Traps to avoid

1. `mcp_manager.py`'s `cwd=_PROJECT_ROOT` assumption breaks in a frozen
   bundle — change to data dir / bundle resources first.
2. Relative default `data/` inside a read-only signed `.app` — the
   `platformdirs` change is mandatory, not optional.
