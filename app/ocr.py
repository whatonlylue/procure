"""Optional OCR for scanned PDFs and image files via the Tesseract 5 CLI.

No Python OCR dependencies: the ``tesseract`` binary is discovered lazily
and callers degrade gracefully when it is missing. Images are passed to
the CLI directly; PDF pages are rendered by ``pypdfium2`` and piped to
Tesseract as binary PGM (no PIL).

Binary lookup order: ``PROCURE_TESSERACT`` (absolute path, set by the
desktop app to its bundled copy together with ``TESSDATA_PREFIX``), then
``PATH``, then common install locations. Language: ``PROCURE_OCR_LANG``
(default ``eng``).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from functools import lru_cache

IMAGE_EXTENSIONS = frozenset(
    {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
)

# Pages with fewer non-whitespace chars than this count as "thin" (likely
# scanned) and become OCR candidates in auto mode.
THIN_PAGE_CHARS = 40

INSTALL_HINT = (
    "OCR needs Tesseract: install it (macOS: brew install tesseract; "
    "Windows: the UB-Mannheim installer; Linux: apt install tesseract-ocr) "
    "or set PROCURE_TESSERACT"
)

_COMMON_PATHS = (
    "/opt/homebrew/bin/tesseract",
    "/usr/local/bin/tesseract",
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
)
_TIMEOUT_S = 120


def _executable(path: str | None) -> bool:
    return bool(path) and os.path.isfile(path) and os.access(path, os.X_OK)


@lru_cache(maxsize=1)
def _find_binary() -> str:
    env = os.environ.get("PROCURE_TESSERACT", "").strip()
    if _executable(env):
        return env
    found = shutil.which("tesseract")
    if found and _executable(found):
        return found
    for p in _COMMON_PATHS:
        if _executable(p):
            return p
    return ""


def reset_cache() -> None:
    """Forget the discovered binary (tests, or env changed at runtime)."""
    _find_binary.cache_clear()


def backend_name() -> str:
    """OCR backend in use: "tesseract" or "" (not installed)."""
    return "tesseract" if _find_binary() else ""


def available() -> bool:
    return backend_name() != ""


def _run(input_arg: str, data: bytes | None = None) -> str:
    binary = _find_binary()
    if not binary:
        raise RuntimeError(INSTALL_HINT)
    lang = os.environ.get("PROCURE_OCR_LANG", "").strip() or "eng"
    cmd = [binary, input_arg, "stdout", "-l", lang, "--psm", "3"]
    kwargs: dict = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    proc = subprocess.run(
        cmd, input=data, capture_output=True, timeout=_TIMEOUT_S, **kwargs
    )
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace").strip()
        last = err.splitlines()[-1] if err else f"exit code {proc.returncode}"
        raise RuntimeError(f"tesseract failed: {last}")
    return proc.stdout.decode("utf-8", errors="replace").strip()


def _ocr_bytes(data: bytes, suffix: str = "") -> str:
    # Close before invoking so Windows can open it; delete afterwards.
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(data)
        name = f.name
    try:
        return _run(name)
    finally:
        try:
            os.unlink(name)
        except OSError:
            pass


def ocr_image(source) -> str:
    """OCR one image (path or bytes) to text. Raises RuntimeError if no backend."""
    if not available():
        raise RuntimeError(INSTALL_HINT)
    if isinstance(source, (bytes, bytearray)):
        return _ocr_bytes(bytes(source))
    return _run(os.fspath(source))


def _pgm_bytes(bitmap) -> bytes:
    """Encode a grayscale pdfium bitmap as binary PGM (P5)."""
    arr = bitmap.to_numpy()
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    h, w = arr.shape[:2]  # to_numpy() already excludes any row-stride padding
    return b"P5\n%d %d\n255\n" % (w, h) + arr.tobytes()


def ocr_pdf_pages(path: str, pages: list[int], dpi: int = 200) -> dict[int, str]:
    """OCR selected 0-based PDF pages (rendered with pypdfium2)."""
    if not available():
        raise RuntimeError(INSTALL_HINT)
    import pypdfium2 as pdfium

    out: dict[int, str] = {}
    doc = pdfium.PdfDocument(path)
    try:
        n = len(doc)
        for pno in pages:
            if pno < 0 or pno >= n:
                continue
            page = doc[pno]
            try:
                bitmap = page.render(scale=dpi / 72, grayscale=True)
                try:
                    pgm = _pgm_bytes(bitmap)
                finally:
                    bitmap.close()
            finally:
                page.close()
            try:
                out[pno] = _run("stdin", pgm)
            except Exception:
                out[pno] = ""
    finally:
        doc.close()
    return out
