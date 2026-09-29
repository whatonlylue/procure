"""Text extraction for supported file types. Stdlib + pypdf only."""
from __future__ import annotations

import os
import re
import zipfile

SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".pdf", ".docx", ".pptx", ".html", ".htm"}


def is_supported(filename: str) -> bool:
    return os.path.splitext(filename.lower())[1] in SUPPORTED_EXTENSIONS


def extract_text(path: str, filename: str) -> str:
    ext = os.path.splitext(filename.lower())[1]
    if ext in (".txt", ".md", ".markdown"):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return _strip_boilerplate(f.read())
    if ext == ".pdf":
        return _extract_pdf(path)
    if ext == ".docx":
        return _extract_docx(path)
    if ext == ".pptx":
        return _extract_pptx(path)
    if ext in (".html", ".htm"):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return _strip_tags(f.read())
    raise ValueError(f"Unsupported file type: {ext or '(none)'}")


def _extract_pdf(path: str) -> str:
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = [(p.extract_text() or "") for p in reader.pages]
    return "\n\n".join(t.strip() for t in pages if t.strip())


def _extract_docx(path: str) -> str:
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml").decode("utf-8", errors="ignore")
    xml = re.sub(r"</w:p[^>]*>", "\n\n", xml)
    return _strip_tags(xml)


def _extract_pptx(path: str) -> str:
    texts: list[str] = []
    with zipfile.ZipFile(path) as z:
        for name in sorted(z.namelist()):
            if re.fullmatch(r"ppt/slides/slide\d+\.xml", name):
                xml = z.read(name).decode("utf-8", errors="ignore")
                texts.append(_strip_tags(xml))
    return "\n\n".join(texts)


_GB_START = "*** START OF THE PROJECT GUTENBERG EBOOK"
_GB_END = "*** END OF THE PROJECT GUTENBERG EBOOK"


def _strip_boilerplate(text: str) -> str:
    """Drop Project Gutenberg license header/footer when present.

    Only applies to texts carrying both markers; everything else is
    returned unchanged.
    """
    s = text.find(_GB_START)
    e = text.find(_GB_END)
    if s == -1 or e == -1 or e <= s:
        return text
    body = text[text.index("\n", s) + 1:e].strip()
    return body or text


def _strip_tags(xml: str) -> str:
    text = re.sub(r"<[^>]+>", " ", xml)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()
