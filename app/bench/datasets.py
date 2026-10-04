"""Benchmark dataset loading: procure JSONL plus a BEIR-style layout.

Procure-native format (offline, human-editable):

- ``corpus.jsonl`` — one doc per line:
  ``{"doc_id": "lighthouse", "filename": "lighthouse.md", "text": "..."}``
  (``filename`` defaults to ``<doc_id>.md``; ``tags``/``doc_type`` optional)
- ``queries.jsonl`` — one query per line:
  ``{"query": "...", "relevant_doc_ids": ["lighthouse"]}``

Each query may instead carry ``relevant_chunk_ids`` to score chunk-level
recall (chunk ids look like ``<doc_id>:<n>`` after ingest). At least one
of the two relevance lists must be non-empty.

BEIR-style directories (``--beir-dir``) hold ``corpus.jsonl`` (``_id`` /
``text`` / optional ``title``), ``queries.jsonl`` or ``queries.tsv``
(``_id`` / ``text``), and ``qrels.tsv`` (``query-id corpus-id score``,
score > 0 counts as relevant). The loader converts them to the procure
format above, so BEIR subsets (e.g. SciFact, FiQA, NFCorpus) can run
through the same runner after ingest.
"""
from __future__ import annotations

import csv
import json
import os


def _read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError as e:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {e}")
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{lineno}: want a JSON object per line")
            rows.append(obj)
    return rows


def load_corpus_jsonl(path: str) -> list[dict]:
    """Load and validate a procure ``corpus.jsonl``."""
    docs = []
    for row in _read_jsonl(path):
        doc_id = str(row.get("doc_id") or "").strip()
        text = row.get("text") or ""
        if not doc_id:
            raise ValueError(f"{path}: corpus row without doc_id: {row!r:.80}")
        if not text.strip():
            raise ValueError(f"{path}: corpus doc {doc_id!r} has no text")
        docs.append({
            "doc_id": doc_id,
            "filename": str(row.get("filename") or f"{doc_id}.md"),
            "text": text,
            "tags": list(row.get("tags") or []),
            "doc_type": str(row.get("doc_type") or "document"),
        })
    if not docs:
        raise ValueError(f"{path}: no corpus documents found")
    ids = [d["doc_id"] for d in docs]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate corpus doc_ids")
    return docs


def load_queries_jsonl(path: str) -> list[dict]:
    """Load and validate a procure ``queries.jsonl``."""
    queries = []
    for row in _read_jsonl(path):
        query = str(row.get("query") or "").strip()
        doc_ids = [str(x) for x in (row.get("relevant_doc_ids") or []) if str(x).strip()]
        chunk_ids = [str(x) for x in (row.get("relevant_chunk_ids") or []) if str(x).strip()]
        if not query:
            raise ValueError(f"{path}: query row without query text: {row!r:.80}")
        if not doc_ids and not chunk_ids:
            raise ValueError(f"{path}: query without relevant ids: {query!r:.60}")
        queries.append({
            "query": query,
            "relevant_doc_ids": doc_ids,
            "relevant_chunk_ids": chunk_ids,
        })
    if not queries:
        raise ValueError(f"{path}: no queries found")
    return queries


def load_beir_dir(path: str) -> tuple[list[dict], list[dict]]:
    """Convert a BEIR-style directory to (corpus, queries) procure rows."""
    corpus_path = os.path.join(path, "corpus.jsonl")
    if not os.path.isfile(corpus_path):
        raise ValueError(f"{path}: missing corpus.jsonl")
    corpus = []
    for row in _read_jsonl(corpus_path):
        cid = str(row.get("_id") or "").strip()
        text = str(row.get("text") or "")
        title = str(row.get("title") or "").strip()
        if not cid or not text.strip():
            continue
        body = f"{title}\n\n{text}" if title else text
        corpus.append({"doc_id": cid, "filename": f"{cid}.md",
                       "text": body, "tags": [], "doc_type": "document"})

    queries_path = os.path.join(path, "queries.jsonl")
    queries_tsv = os.path.join(path, "queries.tsv")
    qtext: dict[str, str] = {}
    if os.path.isfile(queries_path):
        for row in _read_jsonl(queries_path):
            qid = str(row.get("_id") or "").strip()
            if qid and str(row.get("text") or "").strip():
                qtext[qid] = str(row["text"])
    elif os.path.isfile(queries_tsv):
        with open(queries_tsv, encoding="utf-8") as f:
            for row in csv.reader(f, delimiter="\t"):
                if len(row) >= 2 and row[0].strip() and row[1].strip():
                    qtext[row[0].strip()] = row[1].strip()
    else:
        raise ValueError(f"{path}: missing queries.jsonl or queries.tsv")

    qrels_path = os.path.join(path, "qrels.tsv")
    if not os.path.isfile(qrels_path):
        raise ValueError(f"{path}: missing qrels.tsv")
    rel: dict[str, set[str]] = {}
    with open(qrels_path, encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) < 3:
                continue
            try:
                score = float(row[2])
            except ValueError:
                continue
            if score > 0 and row[0].strip() and row[1].strip():
                rel.setdefault(row[0].strip(), set()).add(row[1].strip())

    queries = [{"query": qtext[qid], "relevant_doc_ids": sorted(cids),
                "relevant_chunk_ids": []}
               for qid, cids in rel.items() if qid in qtext]
    if not queries:
        raise ValueError(f"{path}: no queries with qrels survived the join")
    return corpus, queries


SAMPLE_CORPUS = [
    {
        "doc_id": "lighthouse",
        "filename": "lighthouse.md",
        "text": ("Lighthouse maintenance log. The north beacon was relamped "
                 "in March. Spare lenses are stored in the cliff shed. " * 12),
        "tags": [],
        "doc_type": "document",
    },
    {
        "doc_id": "harbor-tides",
        "filename": "harbor-tides.md",
        "text": ("Harbor tide tables for April. High water at dawn near the "
                 "east pier. Ferry departures follow the morning tide. " * 12),
        "tags": [],
        "doc_type": "document",
    },
    {
        "doc_id": "bakery-ovens",
        "filename": "bakery-ovens.md",
        "text": ("Bakery oven calibration notes. The deck oven runs hot on "
                 "the left side. Sourdough proofs for six hours. " * 12),
        "tags": [],
        "doc_type": "document",
    },
]

SAMPLE_QUERIES = [
    {"query": "When was the north beacon relamped?",
     "relevant_doc_ids": ["lighthouse"], "relevant_chunk_ids": []},
    {"query": "Where are spare lenses stored?",
     "relevant_doc_ids": ["lighthouse"], "relevant_chunk_ids": []},
    {"query": "When is high water near the east pier?",
     "relevant_doc_ids": ["harbor-tides"], "relevant_chunk_ids": []},
    {"query": "How long does sourdough proof?",
     "relevant_doc_ids": ["bakery-ovens"], "relevant_chunk_ids": []},
]


def write_sample_dataset(directory: str) -> dict:
    """Write the bundled 3-doc / 4-query smoke dataset; returns paths."""
    os.makedirs(directory, exist_ok=True)
    corpus_path = os.path.join(directory, "corpus.jsonl")
    queries_path = os.path.join(directory, "queries.jsonl")
    with open(corpus_path, "w", encoding="utf-8") as f:
        for doc in SAMPLE_CORPUS:
            f.write(json.dumps(doc) + "\n")
    with open(queries_path, "w", encoding="utf-8") as f:
        for q in SAMPLE_QUERIES:
            f.write(json.dumps(q) + "\n")
    return {"corpus": corpus_path, "queries": queries_path}
