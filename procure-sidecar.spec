# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller onefile build for the procure Tauri sidecar (app/cli.py).

Build (from the project root, after `npm run build` in frontend/):

    .venv/bin/pyinstaller procure-sidecar.spec

Output: dist/procure-sidecar — copy to
src-tauri/binaries/procure-sidecar-<target-triple> (see packaging script).

The neural stack (torch, sentence-transformers, transformers, chromadb) is
deliberately excluded: the app runs fully offline on hash embeddings +
BM25 + heuristic rerank, and every neural import in app/ degrades
gracefully. This keeps the binary ~100MB instead of ~2GB.
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
        "torch",
        "sentence_transformers",
        "transformers",
        "huggingface_hub",
        "tokenizers",
        "safetensors",
        "chromadb",
        "onnxruntime",
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
    name="procure-sidecar",
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
