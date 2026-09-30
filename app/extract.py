"""Text extraction for supported file types.

Dependency-free by default: OOXML (docx/pptx/xlsx), ODF (odt/ods/odp),
and EPUB are all ZIP + XML under the hood, so they parse with stdlib —
the same approach as the original docx/pptx extractors. Scanned PDFs
and images need the optional ``ocr`` extra (see app/ocr.py).

Legacy binary Office files (.doc/.xls/.ppt) are detected and rejected
with conversion guidance: they need OLE parsing (or LibreOffice), which
is out of scope for the offline default.
"""
from __future__ import annotations

import csv
import io
import json
import os
import re
import zipfile

from app.ocr import IMAGE_EXTENSIONS

# Plain-text-ish types read directly.
_TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".rst", ".org", ".tex", ".textile",
    ".log", ".nfo",
}
SUPPORTED_EXTENSIONS = (
    _TEXT_EXTENSIONS
    | {".pdf", ".docx", ".pptx", ".html", ".htm", ".xml"}
    | {".csv", ".tsv", ".xlsx", ".xlsm", ".xltx"}
    | {".epub", ".odt", ".ods", ".odp", ".rtf", ".json"}
    | set(IMAGE_EXTENSIONS)
)

_LEGACY_OFFICE = {
    ".doc": ".docx",
    ".xls": ".xlsx",
    ".ppt": ".pptx",
}

# Caps keep pathological spreadsheets/feeds from exploding memory.
_MAX_ROWS = 5000
_MAX_JSON_STRINGS = 10000


def is_supported(filename: str) -> bool:
    return os.path.splitext(filename.lower())[1] in SUPPORTED_EXTENSIONS


def extract_text(path: str, filename: str, *, ocr_mode: str = "auto") -> str:
    ext = os.path.splitext(filename.lower())[1]
    if ext in _TEXT_EXTENSIONS:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return _strip_boilerplate(f.read())
    if ext == ".pdf":
        return _extract_pdf(path, ocr_mode)
    if ext == ".docx":
        return _extract_docx(path)
    if ext == ".pptx":
        return _extract_pptx(path)
    if ext in (".html", ".htm", ".xml"):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return _strip_tags(f.read())
    if ext in (".csv", ".tsv"):
        return _extract_csv(path, ext)
    if ext in (".xlsx", ".xlsm", ".xltx"):
        return _extract_xlsx(path)
    if ext == ".epub":
        return _extract_epub(path)
    if ext in (".odt", ".ods", ".odp"):
        return _extract_odf(path)
    if ext == ".rtf":
        return _extract_rtf(path)
    if ext == ".json":
        return _extract_json(path)
    if ext in IMAGE_EXTENSIONS:
        return _extract_image(path, ocr_mode)
    if ext in _LEGACY_OFFICE:
        raise ValueError(
            f"Legacy {ext} files need converting — re-save as "
            f"{_LEGACY_OFFICE[ext]} (Word/Excel/LibreOffice) and retry"
        )
    raise ValueError(f"Unsupported file type: {ext or '(none)'}")


# -- pdf (+ scanned-page OCR) -----------------------------------------------


def _extract_pdf(path: str, ocr_mode: str) -> str:
    from pypdf import PdfReader

    from app.ocr import THIN_PAGE_CHARS, available, ocr_pdf_pages

    reader = PdfReader(path)
    pages = [(p.extract_text() or "") for p in reader.pages]
    texts = [t.strip() for t in pages]
    thin = [i for i, t in enumerate(texts)
            if len(re.sub(r"\s+", "", t)) < THIN_PAGE_CHARS]
    if thin and ocr_mode != "off" and available():
        try:
            ocr = ocr_pdf_pages(path, thin)
        except Exception:  # noqa: BLE001 - OCR failure keeps pdftotext
            ocr = {}
        for i, text in ocr.items():
            if text.strip():
                texts[i] = f"{texts[i]}\n\n{text.strip()}".strip()
    body = "\n\n".join(t for t in texts if t)
    if not body and thin and len(pages) > 0 and not available():
        raise ValueError(
            "No text found — this looks like a scanned PDF. "
            "Install OCR support (uv pip install -e '.[ocr]') and retry"
        )
    return body


def _extract_image(path: str, ocr_mode: str) -> str:
    from app.ocr import available, ocr_image

    if ocr_mode == "off" or not available():
        raise ValueError(
            "Image files need OCR support (uv pip install -e '.[ocr]')"
        )
    return ocr_image(path)


# -- docx / pptx (unchanged OOXML zip parsing) -------------------------------


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


# -- csv / tsv (stdlib) ------------------------------------------------------


def _extract_csv(path: str, ext: str) -> str:
    with open(path, "r", encoding="utf-8-sig", errors="ignore", newline="") as f:
        sample = f.read(8192)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
        except csv.Error:
            dialect = csv.excel_tab if ext == ".tsv" else csv.excel
        rows = [[c.strip() for c in row] for row in csv.reader(f, dialect)]
    lines = [" | ".join(r).strip(" |") for r in rows]
    lines = [ln for ln in lines if ln][: _MAX_ROWS]
    return "\n".join(lines)


# -- xlsx (stdlib zip + xml, same spirit as docx/pptx) -----------------------


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _col_index(cell_ref: str) -> int:
    letters = "".join(c for c in cell_ref if c.isalpha()).upper()
    idx = 0
    for ch in letters:
        idx = idx * 26 + (ord(ch) - ord("A") + 1)
    return idx - 1


def _extract_xlsx(path: str) -> str:
    import xml.etree.ElementTree as ET

    with zipfile.ZipFile(path) as z:
        names = set(z.namelist())
        shared: list[str] = []
        if "xl/sharedStrings.xml" in names:
            root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in root.iter():
                if _local(si.tag) == "si":
                    shared.append("".join(
                        (t.text or "") for t in si.iter()
                        if _local(t.tag) == "t"))
        # Sheet names in workbook order -> worksheet paths via rels.
        try:
            wb = ET.fromstring(z.read("xl/workbook.xml"))
        except KeyError:
            return ""
        try:
            rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        except KeyError:
            rels = None
        rel_target = {}
        if rels is not None:
            for el in rels.iter():
                if _local(el.tag) == "Relationship":
                    rel_target[el.get("Id", "")] = el.get("Target", "")
        sheets: list[tuple[str, str]] = []
        for el in wb.iter():
            if _local(el.tag) != "sheet":
                continue
            name = el.get("name", "Sheet")
            rid = next((v for k, v in el.attrib.items() if _local(k) == "id"), "")
            target = rel_target.get(rid, "")
            target = target.split("/")[-1]
            sheets.append((name, f"xl/worksheets/{target}"))
        if not sheets:  # minimal files: worksheets in name order
            sheets = [(f"Sheet{i + 1}", n) for i, n in
                      enumerate(sorted(n for n in names
                                       if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", n)))]
        out: list[str] = []
        for name, ws_path in sheets:
            if ws_path not in names:
                continue
            root = ET.fromstring(z.read(ws_path))
            rows: dict[int, dict[int, str]] = {}
            for row in root.iter():
                if _local(row.tag) != "row":
                    continue
                try:
                    rno = int(row.get("r", "0"))
                except ValueError:
                    continue
                for c in row:
                    if _local(c.tag) != "c":
                        continue
                    ctype = c.get("t", "")
                    val = ""
                    if ctype == "s":
                        for v in c:
                            if _local(v.tag) == "v":
                                try:
                                    val = shared[int((v.text or "").strip())]
                                except (ValueError, IndexError):
                                    val = ""
                    elif ctype == "inlineStr":
                        val = "".join((t.text or "") for t in c.iter()
                                       if _local(t.tag) == "t")
                    else:
                        for v in c:
                            if _local(v.tag) == "v":
                                val = (v.text or "").strip()
                    val = val.strip()
                    if val:
                        rows.setdefault(rno, {})[_col_index(c.get("r", "A"))] = val
            lines = []
            for rno in sorted(rows)[: _MAX_ROWS]:
                cells = rows[rno]
                width = max(cells) + 1 if cells else 0
                lines.append(" | ".join(cells.get(i, "") for i in range(width)).rstrip(" |"))
            body = "\n".join(ln for ln in lines if ln.strip())
            if body:
                out.append(f"# {name}\n{body}")
        return "\n\n".join(out)


# -- epub (stdlib zip + spine order) -----------------------------------------


def _extract_epub(path: str) -> str:
    with zipfile.ZipFile(path) as z:
        names = set(z.namelist())
        opf_path = "content.opf"
        if "META-INF/container.xml" in names:
            container = z.read("META-INF/container.xml").decode("utf-8", errors="ignore")
            m = re.search(r'full-path="([^"]+)"', container)
            if m:
                opf_path = m.group(1)
        if opf_path not in names:
            # Fallback: every xhtml/html file in the archive.
            docs = sorted(n for n in names
                          if n.lower().endswith((".xhtml", ".html", ".htm")))
            return "\n\n".join(_strip_tags(
                z.read(n).decode("utf-8", errors="ignore")) for n in docs).strip()
        base = opf_path.rsplit("/", 1)[0] + "/" if "/" in opf_path else ""
        opf = z.read(opf_path).decode("utf-8", errors="ignore")
        title = ""
        m = re.search(r"<dc:title[^>]*>(.*?)</dc:title>", opf, re.DOTALL)
        if m:
            title = _strip_tags(m.group(1))
        items = {}
        for m in re.finditer(r"<item\s+[^>]*>", opf):
            tag = m.group(0)
            idm = re.search(r'id="([^"]+)"', tag)
            hrefm = re.search(r'href="([^"]+)"', tag)
            if idm and hrefm:
                items[idm.group(1)] = hrefm.group(1)
        spine = [m.group(1) for m in
                 re.finditer(r'<itemref[^>]*idref="([^"]+)"', opf)]
        if not spine:
            spine = list(items)
        sections = []
        for idref in spine:
            href = items.get(idref, "")
            if not href or not href.lower().endswith((".xhtml", ".html", ".htm")):
                continue
            full = base + href
            if full not in names:
                continue
            raw = z.read(full).decode("utf-8", errors="ignore")
            raw = re.sub(r"</(h[1-6]|p)[^>]*>", "\n\n", raw, flags=re.IGNORECASE)
            text = _strip_tags(raw)
            if text:
                sections.append(text)
        body = "\n\n".join(sections)
        return f"# {title}\n\n{body}".strip() if title else body


# -- odf: odt / ods / odp (stdlib zip + content.xml) -------------------------


def _extract_odf(path: str) -> str:
    with zipfile.ZipFile(path) as z:
        xml = z.read("content.xml").decode("utf-8", errors="ignore")
    xml = re.sub(r"</text:h[^>]*>", "\n\n# ", xml)
    xml = re.sub(r"</text:p[^>]*>", "\n\n", xml)
    xml = re.sub(r"</table:table-row[^>]*>", "\n", xml)
    xml = re.sub(r"</table:table-cell[^>]*>", " | ", xml)
    xml = re.sub(r"<text:(tab|line-break)[^>]*/>", " ", xml)
    return _strip_tags(xml)


# -- rtf (small stdlib control-word stripper) --------------------------------


_SKIP_GROUPS = {"fonttbl", "colortbl", "stylesheet", "info", "pict"}


def _extract_rtf(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()
    out: list[str] = []
    i, n = 0, len(raw)
    skip_depth = 0
    depth = 0
    while i < n:
        ch = raw[i]
        if ch == "{":
            # Look ahead for a skip-group destination.
            m = re.match(r"\{\\[*]?([a-z]+)", raw[i:i + 20])
            depth += 1
            if m and m.group(1) in _SKIP_GROUPS and skip_depth == 0:
                skip_depth = depth
            i += 1
            continue
        if ch == "}":
            if skip_depth == depth:
                skip_depth = 0
            depth = max(0, depth - 1)
            i += 1
            continue
        if skip_depth:
            i += 1
            continue
        if ch == "\\":
            m = re.match(r"\\([a-z]+)(-?\d+)?[ ]?", raw[i:])
            if m:
                word = m.group(1)
                if word in ("par", "line"):
                    out.append("\n")
                elif word == "tab":
                    out.append(" ")
                elif word == "u" and m.group(2):
                    try:
                        out.append(chr(int(m.group(2)) % 65536))
                    except ValueError:
                        pass
                    # \uN is followed by one ANSI fallback char; skip it.
                    i += len(m.group(0))
                    if i < n and raw[i] not in ("\\", "{", "}"):
                        i += 1
                    continue
                i += len(m.group(0))
                continue
            m = re.match(r"\\'([0-9a-fA-F]{2})", raw[i:])
            if m:
                out.append(bytes([int(m.group(1), 16)]).decode("latin1"))
                i += 4
                continue
            if i + 1 < n and raw[i + 1] in ("\\", "{", "}"):
                out.append(raw[i + 1])
                i += 2
                continue
            i += 1
            continue
        if ch in ("\r", "\n"):
            i += 1
            continue
        out.append(ch)
        i += 1
    text = "".join(out)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n\s*\n\s*\n+", "\n\n", text).strip()


# -- json --------------------------------------------------------------------


def _extract_json(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return raw.strip()
    strings: list[str] = []

    def walk(node) -> None:
        if len(strings) >= _MAX_JSON_STRINGS:
            return
        if isinstance(node, str):
            if node.strip():
                strings.append(node.strip())
        elif isinstance(node, dict):
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(data)
    return "\n\n".join(strings) if strings else raw.strip()


# -- shared helpers ----------------------------------------------------------


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
