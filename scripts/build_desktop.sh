#!/usr/bin/env bash
# Build the procure desktop app: frontend -> sidecar -> Tauri bundle.
# Run from the project root. Requires: node, a Python env with pyinstaller
# installed, and the Rust toolchain + tauri-cli.
#
# Outputs (per OS):
#   macOS:   src-tauri/target/release/bundle/macos/procure.app (+ .dmg)
#   Windows: src-tauri/target/release/bundle/nsis/procure-*-setup.exe
#   Linux:   src-tauri/target/release/bundle/appimage/*.AppImage
set -euo pipefail
cd "$(dirname "$0")/.."

OS="$(uname -s)"
EXE=""
BUNDLES="app"
case "$OS" in
  Darwin) BUNDLES="app,dmg" ;;
  MINGW*|MSYS*|CYGWIN*|Windows_NT) EXE=".exe"; BUNDLES="nsis" ;;
esac

if [ -n "${VIRTUAL_ENV:-}" ]; then
  PY="$VIRTUAL_ENV/bin/python"
elif [ -x .venv/bin/python ]; then
  PY=".venv/bin/python"
elif [ -x .venv/Scripts/python.exe ]; then
  PY=".venv/Scripts/python.exe"
else
  PY="python"
fi

echo "==> frontend (version $(node -p "require('./frontend/package.json').version"))"
(cd frontend && npm run build)

echo "==> sidecar"
"$PY" -m PyInstaller --noconfirm procure-sidecar.spec

echo "==> stage sidecar for Tauri"
TRIPLE="$(rustc -vV | sed -n 's/^host: //p')"
mkdir -p src-tauri/binaries
cp "dist/procure-sidecar${EXE}" "src-tauri/binaries/procure-sidecar-${TRIPLE}${EXE}"

echo "==> Tauri bundle"
cargo tauri build --bundles "$BUNDLES"

echo "done."
