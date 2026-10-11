#!/usr/bin/env bash
# Stage a self-contained Tesseract 5 (macOS) into src-tauri/resources/tesseract/
# so the desktop app can OCR scanned PDFs with nothing installed.
#
# Prerequisite: Homebrew `tesseract` (brew install tesseract).
# Layout produced:
#   tesseract            real binary, rpaths rewritten to @executable_path/lib
#   lib/*.dylib          every non-system dylib, transitively, @loader_path links
#   tessdata/eng.traineddata   (tessdata_fast, sha256-verified)
#   LICENSES/            licenses of tesseract + every bundled library
# Idempotent: wipes and re-stages each run.
set -euo pipefail
cd "$(dirname "$0")/.."

# Pinned tessdata_fast commit + checksum of eng.traineddata (~4 MB).
TESSDATA_COMMIT="87416418657359cb625c412a48b6e1d6d41c29bd"
TESSDATA_SHA256="7d4322bd2a7749724879683fc3912cb542f19906c83bcc1a52132556427170b2"
TESSDATA_URL="https://github.com/tesseract-ocr/tessdata_fast/raw/${TESSDATA_COMMIT}/eng.traineddata"

DEST="src-tauri/resources/tesseract"
SRC="$(command -v tesseract || true)"
if [ -z "$SRC" ]; then
  echo "tesseract not found; run: brew install tesseract" >&2
  exit 1
fi
# Resolve symlinks (brew's bin/tesseract -> Cellar/...).
SRC="$(python3 -c 'import os,sys;print(os.path.realpath(sys.argv[1]))' "$SRC")"

rm -rf "$DEST"
mkdir -p "$DEST/lib" "$DEST/tessdata" "$DEST/LICENSES"
cp "$SRC" "$DEST/tesseract"
chmod u+w "$DEST/tesseract"

# Non-system dependencies of a Mach-O, one install-name per line.
deps_of() {
  otool -L "$1" | tail -n +2 | awk '{print $1}' \
    | grep -v -e '^/usr/lib/' -e '^/System/' || true
}

# Resolve a dependency reference (absolute, @rpath/, @loader_path/) to a real file.
resolve() {
  local ref="$1" from="$2" base
  base="$(basename "$ref")"
  case "$ref" in
    /*) [ -e "$ref" ] && { python3 -c 'import os,sys;print(os.path.realpath(sys.argv[1]))' "$ref"; return; } ;;
  esac
  # @rpath/@loader_path/@executable_path or dangling: search Homebrew lib dirs.
  local d
  for d in "$(dirname "$from")" "$(brew --prefix)/lib" /opt/homebrew/lib /usr/local/lib; do
    if [ -e "$d/$base" ]; then
      python3 -c 'import os,sys;print(os.path.realpath(sys.argv[1]))' "$d/$base"; return
    fi
  done
  echo "cannot resolve $ref (needed by $from)" >&2
  return 1
}

# Breadth-first collection. queue holds real source paths; copies are keyed by basename.
# (bash 3.2 on macOS: no associative arrays, so use a newline list.)
SEEN=""
QUEUE=("$SRC")
SOURCES=()   # real source files of libs, parallel to staged copies
while [ "${#QUEUE[@]}" -gt 0 ]; do
  cur="${QUEUE[0]}"; QUEUE=("${QUEUE[@]:1}")
  while IFS= read -r ref; do
    [ -n "$ref" ] || continue
    # An install-name pointing at itself (dylib id) is not a dependency.
    real="$(resolve "$ref" "$cur")"
    name="$(basename "$ref")"
    case "$SEEN" in *"|$name|"*) continue ;; esac
    SEEN="$SEEN|$name|"
    cp "$real" "$DEST/lib/$name"
    chmod u+w "$DEST/lib/$name"
    SOURCES+=("$real")
    QUEUE+=("$real")
  done < <(deps_of "$cur" | grep -v "/$(basename "$cur")\$" || true)
done

# Rewrite install names.
rewrite() {  # file prefix
  local f="$1" prefix="$2" ref
  while IFS= read -r ref; do
    [ -n "$ref" ] || continue
    install_name_tool -change "$ref" "$prefix/$(basename "$ref")" "$f" 2>/dev/null
  done < <(deps_of "$f")
}
rewrite "$DEST/tesseract" "@executable_path/lib"
for lib in "$DEST"/lib/*.dylib; do
  install_name_tool -id "@loader_path/$(basename "$lib")" "$lib" 2>/dev/null
  rewrite "$lib" "@loader_path"
done

# install_name_tool invalidates signatures; Apple Silicon kills unsigned Mach-O.
for f in "$DEST/tesseract" "$DEST"/lib/*.dylib; do
  codesign --force --sign - "$f" >/dev/null 2>&1
done

# Language data (verified).
curl -fsSL -o "$DEST/tessdata/eng.traineddata" "$TESSDATA_URL"
ACTUAL="$(shasum -a 256 "$DEST/tessdata/eng.traineddata" | awk '{print $1}')"
if [ "$ACTUAL" != "$TESSDATA_SHA256" ]; then
  echo "eng.traineddata sha256 mismatch: $ACTUAL != $TESSDATA_SHA256" >&2
  exit 1
fi

# Licenses: copy what each Homebrew package ships (Cellar/<pkg>/<ver>/).
for real in "$SRC" ${SOURCES[@]+"${SOURCES[@]}"}; do
  case "$real" in
    */Cellar/*)
      pkgdir="$(echo "$real" | sed -E 's#(.*/Cellar/[^/]+/[^/]+)/.*#\1#')"
      pkg="$(basename "$(dirname "$pkgdir")")"
      mkdir -p "$DEST/LICENSES/$pkg"
      find "$pkgdir" -maxdepth 1 -type f \( -iname 'LICENSE*' -o -iname 'COPYING*' -o -iname 'NOTICE*' -o -iname 'AUTHORS*' \) \
        -exec cp {} "$DEST/LICENSES/$pkg/" \; ;;
  esac
done
cat > "$DEST/LICENSES/README.txt" <<'TXT'
This directory bundles Tesseract OCR (Apache-2.0), Leptonica (BSD-2-Clause)
and the shared libraries they depend on, as built by Homebrew. Each
subdirectory holds the license files that the Homebrew package ships.
Language data: tessdata_fast eng.traineddata (Apache-2.0),
https://github.com/tesseract-ocr/tessdata_fast
Sources: https://github.com/tesseract-ocr/tesseract, https://github.com/DanBloomberg/leptonica
TXT
# Homebrew's leptonica keg ships no license file; record the upstream text pointer.
for d in "$DEST"/LICENSES/*/; do
  [ -n "$(ls -A "$d")" ] || echo "BSD-2-Clause; see https://github.com/DanBloomberg/leptonica/blob/master/leptonica-license.txt" > "${d}LICENSE-NOTE.txt"
done

echo "staged $DEST ($(du -sh "$DEST" | awk '{print $1}'))"
