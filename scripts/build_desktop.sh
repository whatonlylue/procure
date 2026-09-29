#!/usr/bin/env bash
# Build the procure desktop app: frontend -> sidecar -> Tauri bundle.
# Run from the project root. Requires: node, uv-managed .venv with
# pyinstaller installed, and the Rust toolchain + tauri-cli.
#
# Output: src-tauri/target/release/bundle/macos/procure.app
# (plus a .dmg when hdiutil is available — see notes in PACKAGING.md)
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> frontend"
(cd frontend && npm run build)

echo "==> sidecar"
.venv/bin/pyinstaller --noconfirm procure-sidecar.spec

echo "==> stage sidecar for Tauri"
TRIPLE="$(rustc -vV | sed -n 's/^host: //p')"
mkdir -p src-tauri/binaries
cp dist/procure-sidecar "src-tauri/binaries/procure-sidecar-${TRIPLE}"

echo "==> Tauri bundle"
cargo tauri build --bundles app

echo "done: src-tauri/target/release/bundle/macos/procure.app"
