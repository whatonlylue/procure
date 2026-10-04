"""Optional OCR for scanned PDFs and image files.

No hard dependencies: backends are probed lazily and the caller degrades
gracefully when none is installed. ``rapidocr`` (pip-only wheels, no
system binary) is preferred; ``tesseract`` (binary + ``pytesseract``) is
the fallback. PDF page rendering needs ``pymupdf``.
Install with: ``uv pip install -e '.[ocr]'``.
"""
from __future__ import annotations

import shutil
from functools import lru_cache

IMAGE_EXTENSIONS = frozenset(
    {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
)

# Pages with fewer non-whitespace chars than this count as "thin" (likely
# scanned) and become OCR candidates in auto mode.
THIN_PAGE_CHARS = 40


@lru_cache(maxsize=1)
def backend_name() -> str:
    """OCR backend in use: "rapidocr" | "tesseract" | "" (none installed)."""
    try:
        import rapidocr_onnxruntime  # noqa: F401

        return "rapidocr"
    except ImportError:
        pass
    try:
        import pytesseract  # noqa: F401

        if shutil.which("tesseract"):
            return "tesseract"
    except ImportError:
        pass
    return ""


def available() -> bool:
    return backend_name() != ""


def _pil_image(source):
    """Open a path or bytes as a greyscale PIL image."""
    from PIL import Image

    if isinstance(source, (bytes, bytearray)):
        import io

        img = Image.open(io.BytesIO(bytes(source)))
    else:
        img = Image.open(source)
    return img.convert("L")


@lru_cache(maxsize=1)
def _rapid_engine():
    from rapidocr_onnxruntime import RapidOCR

    return RapidOCR()


def ocr_image(source) -> str:
    """OCR one image (path or bytes) to text. Raises RuntimeError if no backend."""
    backend = backend_name()
    if backend == "rapidocr":
        import numpy as np

        img = _pil_image(source)
        result, _ = _rapid_engine()(np.asarray(img))
        if not result:
            return ""
        return "\n".join(str(line[1]).strip() for line in result if len(line) > 1)
    if backend == "tesseract":
        import pytesseract

        return (pytesseract.image_to_string(_pil_image(source)) or "").strip()
    raise RuntimeError(
        "OCR is not installed. Run: uv pip install -e '.[ocr]' "
        "(or install tesseract + pytesseract)"
    )


def ocr_pdf_pages(path: str, pages: list[int], dpi: int = 200) -> dict[int, str]:
    """OCR selected 0-based PDF pages. Needs pymupdf + an OCR backend."""
    if not available():
        raise RuntimeError(
            "OCR is not installed. Run: uv pip install -e '.[ocr]'"
        )
    try:
        import fitz
    except ImportError as e:
        raise RuntimeError(
            "PDF OCR needs pymupdf. Run: uv pip install -e '.[ocr]'"
        ) from e
    out: dict[int, str] = {}
    with fitz.open(path) as doc:
        for pno in pages:
            if pno < 0 or pno >= len(doc):
                continue
            pix = doc[pno].get_pixmap(dpi=dpi)
            try:
                out[pno] = ocr_image(pix.tobytes("png"))
            except Exception:
                out[pno] = ""
    return out
