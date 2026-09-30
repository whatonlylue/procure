"""Fetch a URL and extract its readable text.

Extractor preference (researched 2025-2026 evaluations):

- ``trafilatura`` when installed (highest F1 on article extraction),
- ``readability-lxml`` next (Mozilla Readability port),
- otherwise a dependency-free fallback that pulls <title> plus
  paragraph-level blocks.

Only http/https URLs; responses are capped in size and time.
"""
from __future__ import annotations

import html as _html
import re
import urllib.parse
import urllib.request

MAX_BYTES = 6 * 1024 * 1024

_UA = {"User-Agent": "procure/0.2.1 (+local-first personal library)"}

_BLOCK = re.compile(
    r"<(?:p|h[1-6]|li|article|section|blockquote|pre|figcaption|td)[^>]*>"
    r"(.*?)</(?:p|h[1-6]|li|article|section|blockquote|pre|figcaption|td)>",
    re.IGNORECASE | re.DOTALL,
)
_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"[ \t]+")
_TITLE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)


def fetch_url_text(url: str, timeout: float = 20.0) -> tuple[str, str]:
    """Fetch *url* and return (title, text). Raises ValueError/RuntimeError."""
    url = (url or "").strip()
    parts = urllib.parse.urlparse(url)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise ValueError(f"Only http(s) URLs can be ingested: {url!r}")
    req = urllib.request.Request(url, headers=_UA)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            ctype = (resp.headers.get("Content-Type") or "").lower()
            raw = resp.read(MAX_BYTES + 1)
    except ValueError:
        raise
    except Exception as e:  # noqa: BLE001 - network errors surface as-is
        raise RuntimeError(f"Could not fetch {url}: {e}") from e
    if len(raw) > MAX_BYTES:
        raise RuntimeError(f"Page too large (>{MAX_BYTES // 1024 // 1024} MB): {url}")
    if "text/html" not in ctype and "text/" not in ctype and "xml" not in ctype:
        raise RuntimeError(f"Not a text page ({ctype or 'unknown type'}): {url}")
    charset = "utf-8"
    m = re.search(r"charset=([\w-]+)", ctype)
    if m:
        charset = m.group(1)
    html = raw.decode(charset, errors="ignore")
    if "text/html" not in ctype and "xml" not in ctype:
        title = f"{parts.netloc}{parts.path}"[:120]
        return title.strip() or parts.netloc, html.strip()
    title, text = _extract_trafilatura(html)
    if text:
        return title or _fallback_title(html, url), text
    title, text = _extract_readability(html)
    if text:
        return title or _fallback_title(html, url), text
    return _fallback_title(html, url), _fallback_blocks(html)


def _extract_trafilatura(html: str) -> tuple[str, str]:
    try:
        from trafilatura import extract
        from trafilatura import extract_metadata
    except ImportError:
        return "", ""
    try:
        text = extract(html, include_comments=False, include_tables=True) or ""
        title = ""
        try:
            meta = extract_metadata(html)
            title = (getattr(meta, "title", "") or "").strip()
        except Exception:  # noqa: BLE001 - title is best-effort
            pass
        return title, text.strip()
    except Exception:  # noqa: BLE001 - fall through to next extractor
        return "", ""


def _extract_readability(html: str) -> tuple[str, str]:
    try:
        from readability import Document
    except ImportError:
        return "", ""
    try:
        doc = Document(html)
        summary = doc.summary() or ""
        text = _clean_blocks(_BLOCK.findall(summary)) or _clean_text(summary)
        return (doc.title() or "").strip(), text
    except Exception:  # noqa: BLE001 - fall through to fallback
        return "", ""


def _fallback_title(html: str, url: str) -> str:
    m = _TITLE.search(html)
    if m:
        title = _clean_text(m.group(1))
        if title:
            return title[:200]
    parts = urllib.parse.urlparse(url)
    return f"{parts.netloc}{parts.path}"[:120].strip() or url[:120]


def _fallback_blocks(html: str) -> str:
    # Drop scripts/styles/nav boilerplate before block extraction.
    html = re.sub(
        r"<(script|style|nav|header|footer|noscript)[^>]*>.*?"
        r"</\1>",
        " ",
        html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return _clean_blocks(_BLOCK.findall(html))


def _clean_blocks(blocks: list[str]) -> str:
    out = []
    for b in blocks:
        t = _clean_text(b)
        if len(t) >= 20:
            out.append(t)
    return "\n\n".join(out)


def _clean_text(fragment: str) -> str:
    text = _TAG.sub(" ", fragment)
    text = _html.unescape(text)
    text = _WS.sub(" ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()
