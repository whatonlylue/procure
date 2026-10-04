"""Fetch a URL and extract its readable text.

Extractor preference:

- ``trafilatura`` when installed (highest F1 on article extraction),
- ``readability-lxml`` next (Mozilla Readability port),
- otherwise a dependency-free fallback that pulls <title> plus
  paragraph-level blocks.

Only http/https URLs; responses are capped in size and time. Fetches
that resolve to loopback/link-local/private IPs are rejected (SSRF
guard) unless PROCURE_FETCH_ALLOW_PRIVATE=1.
"""
from __future__ import annotations

import html as _html
import ipaddress
import os
import re
import socket
import urllib.parse
import urllib.request

from app.version import __version__

MAX_BYTES = 6 * 1024 * 1024

_UA = {"User-Agent": f"procure/{__version__} (+local-first personal library)"}

# Block-level elements only: the (?=[\\s>/]) guard keeps <p...> from
# matching <picture>/<param>/<path>, and the backreference requires the
# matching close tag (no <article>...</p> mixes).
_BLOCK = re.compile(
    r"<(p|h[1-6]|li|article|section|blockquote|pre|figcaption|td)(?=[\s>/])[^>]*>"
    r"(.*?)</\1\s*>",
    re.IGNORECASE | re.DOTALL,
)
_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"[ \t]+")
_TITLE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_META_CHARSET = re.compile(
    r'<meta[^>]+charset\s*=\s*["\']?\s*([\w-]+)', re.IGNORECASE)


def _allow_private() -> bool:
    return os.environ.get("PROCURE_FETCH_ALLOW_PRIVATE", "0").strip().lower() not in (
        "", "0", "false", "no", "off")


def _assert_public_url(url: str, host: str) -> None:
    """Reject URLs resolving to non-public IPs (SSRF guard).

    The fetch is reachable from agent prompts via procure_add_url, so a
    prompt-injected agent must not be able to pull internal endpoints
    (127.0.0.1, link-local metadata services, LAN hosts) into the library.
    """
    if _allow_private():
        return
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as e:
        raise RuntimeError(f"Could not resolve {host}: {e}") from e
    for info in infos:
        try:
            ip = ipaddress.ip_address(info[4][0])
        except ValueError:
            continue
        if not ip.is_global:
            raise ValueError(
                f"Refusing to fetch {url!r}: {host} resolves to "
                f"non-public IP {ip} (set PROCURE_FETCH_ALLOW_PRIVATE=1 "
                "to allow)")
    # Literal-IP URLs with no usable addrinfo (shouldn't happen): be strict.
    if not infos:
        raise RuntimeError(f"Could not resolve {host}")


def _decode(raw: bytes, ctype: str) -> str:
    charset = "utf-8"
    m = re.search(r"charset=([\w-]+)", ctype)
    if m:
        try:
            "x".encode(m.group(1))
            charset = m.group(1)
        except LookupError:
            charset = "utf-8"  # unknown header charset: fall back, don't crash
    else:
        m = _META_CHARSET.search(raw[:8192].decode("ascii", errors="ignore"))
        if m:
            try:
                "x".encode(m.group(1))
                charset = m.group(1)
            except LookupError:
                pass
    return raw.decode(charset, errors="ignore")


def fetch_url_text(url: str, timeout: float = 20.0) -> tuple[str, str]:
    """Fetch *url* and return (title, text). Raises ValueError/RuntimeError."""
    url = (url or "").strip()
    parts = urllib.parse.urlparse(url)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise ValueError(f"Only http(s) URLs can be ingested: {url!r}")
    host = parts.hostname or ""
    _assert_public_url(url, host)
    req = urllib.request.Request(url, headers=_UA)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            ctype = (resp.headers.get("Content-Type") or "").lower()
            raw = resp.read(MAX_BYTES + 1)
    except ValueError:
        raise
    except Exception as e:  # network errors surface as-is
        raise RuntimeError(f"Could not fetch {url}: {e}") from e
    if len(raw) > MAX_BYTES:
        raise RuntimeError(f"Page too large (>{MAX_BYTES // 1024 // 1024} MB): {url}")
    if "text/" not in ctype and "xml" not in ctype:
        raise RuntimeError(f"Not a text page ({ctype or 'unknown type'}): {url}")
    page = _decode(raw, ctype)
    if "text/html" not in ctype and "xml" not in ctype:
        title = f"{parts.netloc}{parts.path}"[:120]
        return title.strip() or parts.netloc, page.strip()
    title, text = _extract_trafilatura(page)
    if text:
        return title or _fallback_title(page, url), text
    title, text = _extract_readability(page)
    if text:
        return title or _fallback_title(page, url), text
    return _fallback_title(page, url), _fallback_blocks(page)


def _extract_trafilatura(page: str) -> tuple[str, str]:
    try:
        from trafilatura import extract
        from trafilatura import extract_metadata
    except ImportError:
        return "", ""
    try:
        text = extract(page, include_comments=False, include_tables=True) or ""
        title = ""
        try:
            meta = extract_metadata(page)
            title = (getattr(meta, "title", "") or "").strip()
        except Exception:  # title is best-effort
            pass
        return title, text.strip()
    except Exception:  # fall through to next extractor
        return "", ""


def _extract_readability(page: str) -> tuple[str, str]:
    try:
        from readability import Document
    except ImportError:
        return "", ""
    try:
        doc = Document(page)
        summary = doc.summary() or ""
        text = (_clean_blocks([body for _, body in _BLOCK.findall(summary)])
                or _clean_text(summary))
        return (doc.title() or "").strip(), text
    except Exception:  # fall through to fallback
        return "", ""


def _fallback_title(page: str, url: str) -> str:
    m = _TITLE.search(page)
    if m:
        title = _clean_text(m.group(1))
        if title:
            return title[:200]
    parts = urllib.parse.urlparse(url)
    return f"{parts.netloc}{parts.path}"[:120].strip() or url[:120]


def _fallback_blocks(page: str) -> str:
    # Drop scripts/styles/nav boilerplate before block extraction.
    page = re.sub(
        r"<(script|style|nav|header|footer|noscript)[^>]*>.*?"
        r"</\1\s*>",
        " ",
        page,
        flags=re.IGNORECASE | re.DOTALL,
    )
    # findall returns (tag, body) pairs now that the pattern backreferences.
    return _clean_blocks([body for _, body in _BLOCK.findall(page)])


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
