"""Sentence-aware text splitter (dependency-free).

Splits on sentence boundaries first, then packs whole sentences into chunks of
~chunk_size chars with chunk_overlap chars of overlap (whole sentences only,
so chunks never start or end mid-sentence, except hard-sliced overlong
sentences). Drops chunks <50 chars, unless the whole input would vanish
(a very short document is kept as chunk(s) so it stays ingestible).
"""
from __future__ import annotations

import re

_SENT_END = re.compile(r"(?<=[.!?…])[\"'”’)\]]*\s+")
_MIN_CHUNK = 50

# Conservative abbreviation guard: only these (plus single initials like
# "U.S.A.") suppress a sentence split. Generic patterns like [A-Z][a-z]?.
# would also swallow real ends ("No.", "Go.", "Hi.") while missing "Mrs.".
_ABBREVIATIONS = frozenset({
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "st.", "vs.",
    "etc.", "e.g.", "i.e.", "fig.",
})
_INITIALS = re.compile(r"(?:[A-Z]\.)+")


def split_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[str]:
    text = (text or "").replace("\r\n", "\n").strip()
    if not text:
        return []
    sentences = [s for s in _split_sentences(text) if s.strip()]
    chunks = _pack(sentences, chunk_size, chunk_overlap)
    kept = [c for c in chunks if len(c.strip()) >= _MIN_CHUNK]
    if kept:
        return kept
    # Everything fell below the junk-fragment floor: the whole input is a
    # very short document (e.g. a BEIR stub like a disambiguation line).
    # Keep it as chunk(s) so short-but-real content ingests instead of
    # failing downstream with "No text extracted".
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


def _split_sentences(text: str) -> list[str]:
    # Split paragraphs first so the sentence regex never joins across them.
    out: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = re.sub(r"\s+", " ", para).strip()
        if not para:
            continue
        start = 0
        for m in _SENT_END.finditer(para):
            end = m.end()
            # Skip splits after known abbreviations / initials ("Mr. X",
            # "U.S.A."). Strip first: the match ends in whitespace, so an
            # un-stripped rsplit would yield "" and the guard would never
            # fire for space-separated text.
            tail = para[start:end].strip().rsplit(" ", 1)[-1]
            if tail.lower() in _ABBREVIATIONS or _INITIALS.fullmatch(tail):
                continue
            out.append(para[start:end].strip())
            start = end
        out.append(para[start:].strip())
    return out


def _pack(sentences: list[str], chunk_size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for sent in sentences:
        while len(sent) > chunk_size:
            # Overlong sentence: flush current, then hard-slice the sentence.
            if cur:
                chunks.append(" ".join(cur))
                cur, cur_len = [], 0
            chunks.append(sent[:chunk_size])
            sent = sent[chunk_size - overlap:] if overlap < chunk_size else ""
        if not sent:
            continue
        add = len(sent) + (1 if cur else 0)
        if cur and cur_len + add > chunk_size:
            chunks.append(" ".join(cur))
            # Overlap: carry back whole trailing sentences totalling ~overlap.
            kept: list[str] = []
            kept_len = 0
            for s in reversed(cur):
                if kept and kept_len + len(s) > overlap:
                    break
                kept.append(s)
                kept_len += len(s) + 1
            cur = list(reversed(kept))
            cur_len = sum(len(s) for s in cur) + max(0, len(cur) - 1)
            # The carried overlap plus the new sentence must still fit: drop
            # oldest carried sentences first (a lone carried sentence longer
            # than the budget is dropped entirely rather than duplicated).
            add = len(sent) + (1 if cur else 0)
            while cur and cur_len + add > chunk_size:
                dropped = cur.pop(0)
                cur_len -= len(dropped) + (1 if cur else 0)
                add = len(sent) + (1 if cur else 0)
        cur.append(sent)
        cur_len += len(sent) + (1 if len(cur) > 1 else 0)
    if cur:
        chunks.append(" ".join(cur))
    return chunks
