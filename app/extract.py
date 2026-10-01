"""Text extraction for supported file types.

Dependency-free by default: OOXML (docx/pptx/xlsx), ODF (odt/ods/odp),
and EPUB are all ZIP + XML under the hood, so they parse with stdlib.
Scanned PDFs and images need the optional ``ocr`` extra (see app/ocr.py).

Legacy binary Office files (.doc/.xls/.ppt) are detected and rejected
with conversion guidance: they need OLE parsing (or LibreOffice), which
is out of scope for the offline default.
"""
from __future__ import annotations

import csv
import html as _html
import json
import os
import posixpath
import re
import urllib.parse
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

# Output caps for row/string-heavy formats. CSV applies its cap while
# streaming rows; XLSX stops parsing sheet rows past the cap; JSON stops
# collecting strings past the cap. (The archive XML itself still parses
# fully — the caps bound extracted text, not parser input.)
_MAX_ROWS = 5000
_MAX_JSON_STRINGS = 10000


def is_supported(filename: str) -> bool:
    return os.path.splitext(filename.lower())[1] in SUPPORTED_EXTENSIONS


def extract_text(path: str, filename: str, *, ocr_mode: str = "auto") -> str:
    # Display titles (e.g. pasted-text "memory: <topic>") may carry no
    # extension; the raw copy on disk always does, so fall back to it.
    ext = (os.path.splitext(filename.lower())[1]
           or os.path.splitext(path.lower())[1])
    if ext in _TEXT_EXTENSIONS:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return _strip_boilerplate(f.read())
    if ext == ".pdf":
        return _extract_pdf(path, ocr_mode)
    if ext == ".docx":
        return _extract_docx(path)
    if ext == ".pptx":
        return _extract_pptx(path)
    if ext in (".html", ".htm"):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return _strip_tags(_drop_script_style(f.read()))
    if ext == ".xml":
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
    force_all = ocr_mode == "on"
    ocr_targets = list(range(len(pages))) if force_all else thin
    if ocr_targets and ocr_mode != "off" and available():
        try:
            ocr = ocr_pdf_pages(path, ocr_targets)
        except Exception:  # OCR failure keeps pdftotext
            ocr = {}
        for i, text in ocr.items():
            if text.strip():
                texts[i] = f"{texts[i]}\n\n{text.strip()}".strip()
    body = "\n\n".join(t for t in texts if t)
    if not body and thin and pages:
        if not available():
            raise ValueError(
                "No text found — this looks like a scanned PDF. "
                "Install OCR support (uv pip install -e '.[ocr]') and retry"
            )
        if ocr_mode == "off":
            raise ValueError(
                "No text found — this looks like a scanned PDF, but OCR is "
                "disabled (PROCURE_OCR=off). Set it to auto and retry"
            )
        raise ValueError(
            "No text found — this looks like a scanned PDF, but OCR "
            "returned no text for its pages"
        )
    return body


def _extract_image(path: str, ocr_mode: str) -> str:
    from app.ocr import available, ocr_image

    if ocr_mode == "off" or not available():
        raise ValueError(
            "Image files need OCR support (uv pip install -e '.[ocr]')"
        )
    return ocr_image(path)


# -- docx / pptx (OOXML zip parsing, stdlib) ----------------------------------


def _extract_docx(path: str) -> str:
    with zipfile.ZipFile(path) as z:
        names = set(z.namelist())
        parts = [_para_breaks(z.read("word/document.xml")
                              .decode("utf-8", errors="ignore"))]
        # Headers, footers, footnotes, and comments carry real content too.
        for name in sorted(names):
            if re.fullmatch(r"word/(header\d+|footer\d+|footnotes|comments)\.xml",
                            name):
                parts.append(_para_breaks(
                    z.read(name).decode("utf-8", errors="ignore")))
    return "\n\n".join(t for t in (_strip_tags(p) for p in parts) if t)


def _para_breaks(xml: str) -> str:
    # Exact </w:p>: the looser </w:p[^>]*> also matches </w:pPr>.
    return re.sub(r"</w:p\s*>", "\n\n", xml)


def _extract_pptx(path: str) -> str:
    def slide_no(name: str) -> int:
        m = re.fullmatch(r"ppt/slides/slide(\d+)\.xml", name)
        return int(m.group(1)) if m else 1 << 30

    texts: list[str] = []
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist()
                 if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)]
        # Numeric sort: lexicographic order gives slide1, slide10, slide2.
        for name in sorted(names, key=slide_no):
            xml = z.read(name).decode("utf-8", errors="ignore")
            texts.append(_strip_tags(xml))
    return "\n\n".join(texts)


# -- csv / tsv (stdlib) ------------------------------------------------------


def _extract_csv(path: str, ext: str) -> str:
    lines: list[str] = []
    with open(path, "r", encoding="utf-8-sig", errors="ignore", newline="") as f:
        sample = f.read(8192)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
        except csv.Error:
            dialect = csv.excel_tab if ext == ".tsv" else csv.excel
        # Stream: stop reading once the cap is hit, instead of parsing the
        # whole file and truncating afterwards.
        for row in csv.reader(f, dialect):
            line = " | ".join(c.strip() for c in row).strip(" |")
            if not line:
                continue
            lines.append(line)
            if len(lines) >= _MAX_ROWS:
                break
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


# Built-in Excel date/time number formats (serials render as ISO dates).
_DATE_FMT_IDS = frozenset(
    list(range(14, 23)) + list(range(27, 37)) + [45, 46, 47, 50, 57]
)


def _excel_serial_to_iso(serial: float, date1904: bool) -> str:
    import datetime as _dt

    base = _dt.datetime(1904, 1, 1) if date1904 else _dt.datetime(1899, 12, 30)
    try:
        moment = base + _dt.timedelta(days=serial)
    except (OverflowError, ValueError):
        return str(serial)
    if serial % 1:
        return moment.strftime("%Y-%m-%d %H:%M")
    return moment.strftime("%Y-%m-%d")


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
        date1904 = any(_local(el.tag) == "workbookPr"
                       and el.get("date1904") == "1" for el in wb.iter())
        # Column styles: which style indexes render as dates.
        date_styles: set[int] = set()
        try:
            styles = ET.fromstring(z.read("xl/styles.xml"))
            num_fmts: dict[str, str] = {}
            for el in styles.iter():
                if _local(el.tag) == "numFmt":
                    num_fmts[el.get("numFmtId", "")] = el.get("formatCode", "")
            # Only cellXfs entries are addressable via a cell's s attribute
            # (cellStyleXfs indexes are a different list).
            xfs: list = []
            for el in styles.iter():
                if _local(el.tag) == "cellXfs":
                    xfs = [c for c in el if _local(c.tag) == "xf"]
                    break
            for i, xf in enumerate(xfs):
                fmt_id = xf.get("numFmtId", "0")
                try:
                    is_date = int(fmt_id) in _DATE_FMT_IDS
                except ValueError:
                    is_date = False
                if not is_date and fmt_id in num_fmts:
                    code = num_fmts[fmt_id].lower()
                    is_date = any(tok in code for tok in
                                  ("y", "m", "d", "h", "s"))
                if is_date:
                    date_styles.add(i)
        except KeyError:
            pass
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
            lines: list[str] = []
            # iterparse + element clearing keeps memory flat; stop early
            # once the row cap is hit.
            with z.open(ws_path) as fh:
                ctx = ET.iterparse(fh, events=("end",))
                for _, elem in ctx:
                    if _local(elem.tag) != "row":
                        continue
                    cells: dict[int, str] = {}
                    for c in elem:
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
                        elif ctype == "b":
                            raw_b = "".join(v.text or "" for v in c
                                            if _local(v.tag) == "v").strip()
                            val = "FALSE" if raw_b in ("", "0") else "TRUE"
                        elif ctype in ("e", "str", ""):
                            for v in c:
                                if _local(v.tag) == "v":
                                    val = (v.text or "").strip()
                            if ctype == "" and val:
                                try:
                                    style = int(c.get("s", "-1"))
                                except ValueError:
                                    style = -1
                                if style in date_styles:
                                    try:
                                        val = _excel_serial_to_iso(float(val), date1904)
                                    except ValueError:
                                        pass
                        else:
                            for v in c:
                                if _local(v.tag) == "v":
                                    val = (v.text or "").strip()
                        val = val.strip()
                        if val:
                            cells[_col_index(c.get("r", "A"))] = val
                    if cells:
                        width = max(cells) + 1
                        line = (" | ".join(cells.get(i, "") for i in range(width))
                                .rstrip(" |"))
                        if line.strip():
                            lines.append(line)
                            if len(lines) >= _MAX_ROWS:
                                elem.clear()
                                break
                    elem.clear()
            body = "\n".join(lines)
            if body:
                out.append(f"# {name}\n{body}")
        return "\n\n".join(out)


# -- epub (stdlib zip + spine order) -----------------------------------------


_ATTR_RE_CACHE: dict[str, re.Pattern] = {}


def _attr(tag: str, name: str) -> str:
    """Single XML attribute value; accepts single or double quotes."""
    pat = _ATTR_RE_CACHE.get(name)
    if pat is None:
        pat = re.compile(rf"{re.escape(name)}\s*=\s*(['\"])(.*?)\1")
        _ATTR_RE_CACHE[name] = pat
    m = pat.search(tag)
    return m.group(2) if m else ""


def _html_para_breaks(raw: str) -> str:
    return re.sub(r"</(h[1-6]|p)[^>]*>", "\n\n", raw, flags=re.IGNORECASE)


def _extract_epub(path: str) -> str:
    with zipfile.ZipFile(path) as z:
        names = set(z.namelist())
        opf_path = "content.opf"
        if "META-INF/container.xml" in names:
            container = z.read("META-INF/container.xml").decode("utf-8", errors="ignore")
            m = re.search(r'full-path=(["\'])(.*?)\1', container)
            if m:
                opf_path = m.group(2)
        if opf_path not in names:
            # Fallback: every xhtml/html file in the archive.
            docs = sorted(n for n in names
                          if n.lower().endswith((".xhtml", ".html", ".htm")))
            return "\n\n".join(_strip_tags(_html_para_breaks(
                z.read(n).decode("utf-8", errors="ignore"))) for n in docs).strip()
        base = opf_path.rsplit("/", 1)[0] + "/" if "/" in opf_path else ""
        opf = z.read(opf_path).decode("utf-8", errors="ignore")
        title = ""
        m = re.search(r"<dc:title[^>]*>(.*?)</dc:title>", opf, re.DOTALL)
        if m:
            title = _strip_tags(m.group(1))
        items = {}
        for m in re.finditer(r"<item\s+[^>]*>", opf):
            tag = m.group(0)
            item_id = _attr(tag, "id")
            href = _attr(tag, "href")
            if item_id and href:
                items[item_id] = href
        spine = [_attr(m.group(0), "idref")
                 for m in re.finditer(r"<itemref\s+[^>]*>", opf)]
        spine = [s for s in spine if s]
        if not spine:
            spine = list(items)
        sections = []
        for idref in spine:
            href = items.get(idref, "")
            if not href or not href.lower().endswith((".xhtml", ".html", ".htm")):
                continue
            # Hrefs may be URL-encoded (%20) or relative (../Text/ch1.xhtml).
            full = posixpath.normpath(posixpath.join(
                base, urllib.parse.unquote(href)))
            if full not in names:
                continue
            raw = z.read(full).decode("utf-8", errors="ignore")
            text = _strip_tags(_html_para_breaks(raw))
            if text:
                sections.append(text)
        body = "\n\n".join(sections)
        return f"# {title}\n\n{body}".strip() if title else body


# -- odf: odt / ods / odp (stdlib zip + content.xml) -------------------------


def _extract_odf(path: str) -> str:
    with zipfile.ZipFile(path) as z:
        xml = z.read("content.xml").decode("utf-8", errors="ignore")
    # "# " goes at the OPENING heading tag: appending it at the close would
    # prefix the paragraph AFTER the heading instead.
    xml = re.sub(r"<text:h[^>]*>", "# ", xml)
    xml = re.sub(r"</text:h[^>]*>", "\n\n", xml)
    xml = re.sub(r"</text:p[^>]*>", "\n\n", xml)
    xml = re.sub(r"</table:table-row[^>]*>", "\n", xml)
    xml = re.sub(r"</table:table-cell[^>]*>", " | ", xml)
    xml = re.sub(r"<text:(tab|line-break)[^>]*/>", " ", xml)
    return _strip_tags(xml)


# -- rtf (small stdlib control-word stripper) --------------------------------


_SKIP_GROUPS = {"fonttbl", "colortbl", "stylesheet", "info", "pict"}

_CTRL_WORD = re.compile(r"\\([a-z]+)(-?\d+)?[ ]?")
_CTRL_HEX = re.compile(r"\\'([0-9a-fA-F]{2})")
_SKIP_OPEN = re.compile(r"\{\\[*]?([a-z]+)")
_CODEPAGE = re.compile(r"\\ansicpg(\d+)")


def _extract_rtf(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()
    # \'hh bytes use the declared ANSI codepage when present.
    encoding = "latin1"
    m = _CODEPAGE.search(raw[:2000])
    if m:
        try:
            import codecs

            codecs.lookup(f"cp{m.group(1)}")
            encoding = f"cp{m.group(1)}"
        except LookupError:
            pass
    out: list[str] = []
    i, n = 0, len(raw)
    skip_depth = 0
    depth = 0
    while i < n:
        ch = raw[i]
        if ch == "{":
            # Look ahead for a skip-group destination. {\* ...} groups are
            # ignorable by definition (themedata, generator, rsidtbl...).
            m = _SKIP_OPEN.match(raw, i, min(n, i + 24))
            depth += 1
            if skip_depth == 0 and raw[i + 1:i + 3] == "\\*":
                skip_depth = depth
            elif m and m.group(1) in _SKIP_GROUPS and skip_depth == 0:
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
            m = _CTRL_WORD.match(raw, i)
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
            m = _CTRL_HEX.match(raw, i)
            if m:
                out.append(bytes([int(m.group(1), 16)]).decode(
                    encoding, errors="ignore"))
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
    # Iterative walk with a depth cap: recursion would hit RecursionError
    # on deeply nested documents.
    strings: list[str] = []
    stack: list[tuple[object, int]] = [(data, 0)]
    while stack and len(strings) < _MAX_JSON_STRINGS:
        node, depth = stack.pop()
        if isinstance(node, str):
            if node.strip():
                strings.append(node.strip())
        elif depth >= 100:
            continue
        elif isinstance(node, dict):
            stack.extend((v, depth + 1) for v in reversed(list(node.values())))
        elif isinstance(node, list):
            stack.extend((v, depth + 1) for v in reversed(node))
    return "\n\n".join(strings) if strings else raw.strip()


# -- shared helpers ----------------------------------------------------------


def _drop_script_style(page: str) -> str:
    """Remove script/style blocks (code and CSS are not document text)."""
    return re.sub(
        r"<(script|style)[^>]*>.*?</\1\s*>",
        " ",
        page,
        flags=re.IGNORECASE | re.DOTALL,
    )


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
    # Entities (&amp;, &#8217;, &lt; ...) decode AFTER tag stripping so
    # literal "&lt;tag&gt;" text can't turn into a stripped tag.
    text = _html.unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()
