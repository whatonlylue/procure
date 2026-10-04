# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller onefile build for the procure Tauri sidecar (app/cli.py).

Build (from the project root, after `npm run build` in frontend/):

    .venv/bin/pyinstaller procure-sidecar.spec

Output: dist/ProcureHelper — copy to
src-tauri/binaries/ProcureHelper-<target-triple> (see packaging script).

The output name is the user-visible process name: Activity Monitor /
Task Manager show the sidecar (bootloader + payload child, plus two
more when the MCP server re-invokes this same binary) as ProcureHelper
alongside the Procure app shell, so compute-heavy Python work is
attributable at a glance. The MCP server is not an independent binary
(it re-invokes this one via the `mcp-server` subcommand), so it
deliberately shares the ProcureHelper name rather than getting its own.

Embeddings run on ONNX Runtime (granite-embedding-small-english-r2,
app/embeddings.py), so onnxruntime + tokenizers + huggingface_hub ship in
the sidecar and the model (~100MB fp16) downloads once into the HF cache
on first ingest. The torch rerank stack (torch, sentence-transformers,
transformers) stays excluded: every neural import in app/ degrades
gracefully, which keeps torch's ~2GB out of the binary.

The benchmark harness (app.bench) is likewise excluded: it is source-only
(still in git for `git clone` users, but not in wheels either), and
`procure bench` in app/cli.py degrades to a clear error without it.
"""

a = Analysis(
    ["app/cli.py"],
    pathex=[],
    binaries=[],
    datas=[
        ("app/static", "app/static"),
        ("skills", "skills"),
    ],
    hiddenimports=[
        "uvicorn.logging",
        "uvicorn.loops.auto",
        "uvicorn.protocols.http.auto",
        "uvicorn.protocols.websockets.auto",
        "uvicorn.lifespan.on",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "app.bench",
        "app.bench.datasets",
        "app.bench.metrics",
        "app.bench.runner",
        "app.bench.sources",
        "torch",
        "sentence_transformers",
        "transformers",
        "safetensors",
        "sklearn",
        "scipy",
        "pandas",
        "matplotlib",
        "PIL",
        "cv2",
        "tensorflow",
        "pytest",
        "IPython",
        "jupyter",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="ProcureHelper",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # UPX breaks macOS code signing; keep off.
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # Tauri reads PROCURE_URL + logs from stdout.
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
