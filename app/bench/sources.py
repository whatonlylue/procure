"""Download real BEIR/MTEB retrieval datasets into ``data/benchmarks/``.

Provenance (verified against the HuggingFace Hub):

- ``mteb/<repo>`` (default source) keeps the classic BEIR layout —
  ``corpus.jsonl`` (``_id`` / ``title`` / ``text``), ``queries.jsonl``,
  ``qrels/<split>.tsv`` — readable with stdlib alone. These are the BEIR
  datasets repackaged by the MTEB team: same documents, queries, qrels.
- ``BeIR/<Dataset>`` (``--source beir``) stores corpus/queries as
  sharded parquet with qrels in a separate ``BeIR/<Dataset>-qrels`` repo.
  It needs ``pyarrow`` (``pip install pyarrow``); without it pull fails
  with a pointer back at the ``mteb`` source, which carries the same data.

Only converted files are kept (procure ``corpus.jsonl``/``queries.jsonl``
plus the native ``qrels.tsv`` and a ``manifest.json`` with the source
URLs), so a multi-GB corpus is never stored twice. ``data/`` is
git-ignored: pulls never touch the repo.
"""
from __future__ import annotations

import csv
import datetime
import json
import os
import shutil
import sys
import urllib.error
import urllib.request

HF_RESOLVE = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"
HF_API = "https://huggingface.co/api/datasets/{repo}"

# Curated BEIR core: registry key -> hosting. ``mteb`` names are the exact
# Hub repos (verified); ``beir``/``beir_qrels`` are the matching BeIR org
# repos (parquet corpus/queries, TSV qrels). Sizes are approximate
# download sizes of the converted corpus.
REGISTRY: dict[str, dict] = {
    "scifact": {
        "mteb": "mteb/scifact", "beir": "BeIR/scifact",
        "beir_qrels": "BeIR/scifact-qrels",
        "size": "~5K docs", "domain": "fact verification (scientific claims)",
    },
    "nfcorpus": {
        "mteb": "mteb/nfcorpus", "beir": "BeIR/nfcorpus",
        "beir_qrels": "BeIR/nfcorpus-qrels",
        "size": "~3.6K docs", "domain": "biomedical retrieval",
    },
    "fiqa": {
        "mteb": "mteb/fiqa", "beir": "BeIR/fiqa",
        "beir_qrels": "BeIR/fiqa-qrels",
        "size": "~57K docs", "domain": "financial QA",
    },
    "arguana": {
        "mteb": "mteb/arguana", "beir": "BeIR/arguana",
        "beir_qrels": "BeIR/arguana-qrels",
        "size": "~8.7K docs", "domain": "argument retrieval",
    },
    "quora": {
        "mteb": "mteb/quora-retrieval", "beir": "BeIR/quora",
        "beir_qrels": "BeIR/quora-qrels",
        "size": "~520K docs", "domain": "duplicate questions",
    },
    "scidocs": {
        "mteb": "mteb/scidocs", "beir": "BeIR/scidocs",
        "beir_qrels": "BeIR/scidocs-qrels",
        "size": "~25K docs", "domain": "scientific citation prediction",
    },
    "fever": {
        "mteb": "mteb/fever", "beir": "BeIR/fever",
        "beir_qrels": "BeIR/fever-qrels",
        "size": "~5.4M docs", "domain": "fact verification (Wikipedia)",
    },
    "hotpotqa": {
        "mteb": "mteb/hotpotqa", "beir": "BeIR/hotpotqa",
        "beir_qrels": "BeIR/hotpotqa-qrels",
        "size": "~5.2M docs", "domain": "multi-hop QA (Wikipedia)",
    },
    "nq": {
        "mteb": "mteb/nq", "beir": "BeIR/nq",
        "beir_qrels": "BeIR/nq-qrels",
        "size": "~2.7M docs", "domain": "natural questions (Wikipedia)",
    },
    "msmarco": {
        "mteb": "mteb/msmarco", "beir": "BeIR/msmarco",
        "beir_qrels": "BeIR/msmarco-qrels",
        "size": "~8.8M docs", "domain": "web passage ranking",
    },
    "dbpedia": {
        "mteb": "mteb/dbpedia", "beir": "BeIR/dbpedia-entity",
        "beir_qrels": "BeIR/dbpedia-entity-qrels",
        "size": "~4.6M docs", "domain": "entity retrieval",
    },
    "trec-covid": {
        "mteb": "mteb/trec-covid", "beir": "BeIR/trec-covid",
        "beir_qrels": "BeIR/trec-covid-qrels",
        "size": "~170K docs", "domain": "biomedical (COVID-19)",
    },
    "touche2020": {
        "mteb": "mteb/touche2020", "beir": "BeIR/webis-touche2020",
        "beir_qrels": "BeIR/webis-touche2020-qrels",
        "size": "~380K docs", "domain": "argument retrieval (web)",
    },
    "cqadupstack": {
        "mteb": "mteb/cqadupstack-retrieval", "beir": "BeIR/cqadupstack",
        "beir_qrels": "BeIR/cqadupstack-qrels",
        "size": "~460K docs", "domain": "community QA duplicates",
    },
    "climate-fever": {
        "mteb": "mteb/climate-fever", "beir": "BeIR/climate-fever",
        "beir_qrels": "BeIR/climate-fever-qrels",
        "size": "~5.4M docs", "domain": "fact verification (climate)",
    },
}

SPLITS = ("test", "dev", "train")


def list_datasets() -> list[dict]:
    """Registry rows for ``bench list``: key, source repos, size, domain."""
    return [{"dataset": k, **v} for k, v in REGISTRY.items()]


def resolve_repo(dataset: str, source: str) -> str:
    """Map a registry key (or explicit ``org/repo``) to a Hub repo id."""
    if source not in ("mteb", "beir"):
        raise ValueError(f"unknown source {source!r} (want mteb|beir)")
    if "/" in dataset:  # explicit repo passthrough (mteb layout assumed)
        if source != "mteb":
            raise ValueError("--source beir needs a registry dataset key")
        return dataset
    try:
        entry = REGISTRY[dataset]
    except KeyError:
        known = ", ".join(sorted(REGISTRY))
        raise ValueError(f"unknown dataset {dataset!r} (known: {known})")
    return entry[source]


def download_file(url: str, dest: str) -> str:
    """Stream ``url`` to ``dest`` (stdlib; progress on stderr)."""
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "procure-bench/1.0"})
    try:
        resp = urllib.request.urlopen(req)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"download failed ({e.code}): {url}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"download failed ({e.reason}): {url}")
    total = resp.getheader("Content-Length")
    total_mb = f" / {int(total) / 1e6:.1f} MB" if total else ""
    with resp, open(dest, "wb") as f:
        done = 0
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            print(f"\r  {done / 1e6:.1f}{total_mb} MB", end="",
                  flush=True, file=sys.stderr)
    print("", file=sys.stderr)
    return dest


def hf_siblings(repo: str) -> list[str]:
    """File list of a Hub dataset repo (used to find parquet shards)."""
    url = HF_API.format(repo=repo)
    req = urllib.request.Request(url, headers={"User-Agent": "procure-bench/1.0"})
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.load(resp)
    except (urllib.error.URLError, ValueError) as e:
        raise RuntimeError(f"could not list {repo}: {e}")
    return [s.get("rfilename", "") for s in data.get("siblings", [])]


def read_parquet_rows(path: str) -> list[dict]:
    """Read a parquet file into row dicts (needs ``pip install pyarrow``)."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise RuntimeError(
            "reading BeIR parquet needs pyarrow (pip install pyarrow) — "
            "or pull the same data dependency-free with --source mteb")
    return pq.read_table(path).to_pylist()


def _convert_corpus_row(row: dict) -> dict | None:
    cid = str(row.get("_id") or "").strip()
    text = str(row.get("text") or "")
    title = str(row.get("title") or "").strip()
    if not cid or not text.strip():
        return None
    return {"doc_id": cid, "filename": f"{cid}.md",
            "text": f"{title}\n\n{text}" if title else text,
            "tags": [], "doc_type": "document"}


def _read_qrels_tsv(path: str) -> dict[str, set[str]]:
    """BEIR qrels TSV (``query-id corpus-id score``) -> relevant doc ids.

    Skips header lines and zero-scored rows; score > 0 counts as relevant.
    """
    rel: dict[str, set[str]] = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) < 3 or not row[0].strip() or not row[1].strip():
                continue
            try:
                score = float(row[2])
            except ValueError:
                continue  # header line
            if score > 0:
                rel.setdefault(row[0].strip(), set()).add(row[1].strip())
    return rel


def pull(dataset: str, source: str = "mteb", split: str = "test",
         max_queries: int | None = None, dest: str | None = None,
         fetcher=download_file) -> dict:
    """Download a full dataset and convert it to bench-ready files.

    Writes ``corpus.jsonl`` + ``queries.jsonl`` (procure format),
    ``qrels.tsv`` (native copy) and ``manifest.json`` into ``dest``
    (default ``data/benchmarks/<dataset>/``). The corpus is always
    complete — ``max_queries`` only keeps the first N judged queries,
    so a quick smoke run never silently shrinks the retrieval task.
    """
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r} (want test|dev|train)")
    key = dataset.split("/", 1)[-1].lower().replace("_", "-")
    repo = resolve_repo(dataset, source)
    dest = dest or os.path.join("data", "benchmarks", key)
    os.makedirs(dest, exist_ok=True)
    if source == "mteb":
        return _pull_mteb(repo, key, dest, split, max_queries, fetcher)
    return _pull_beir(repo, key, dest, split, max_queries, fetcher)


def _pull_mteb(repo: str, key: str, dest: str, split: str,
               max_queries: int | None, fetcher) -> dict:
    urls = {
        "corpus": HF_RESOLVE.format(repo=repo, path="corpus.jsonl"),
        "queries": HF_RESOLVE.format(repo=repo, path="queries.jsonl"),
        "qrels": HF_RESOLVE.format(repo=repo, path=f"qrels/{split}.tsv"),
    }
    tmp = os.path.join(dest, ".native")
    os.makedirs(tmp, exist_ok=True)
    native_corpus = os.path.join(tmp, "corpus.jsonl")
    native_queries = os.path.join(tmp, "queries.jsonl")
    native_qrels = os.path.join(tmp, "qrels.tsv")
    for label, path in (("corpus", native_corpus), ("queries", native_queries)):
        print(f"downloading {label} ({repo}) ...")
        fetcher(urls[label], path)
    print(f"downloading qrels/{split} ({repo}) ...")
    try:
        fetcher(urls["qrels"], native_qrels)
    except RuntimeError as e:
        raise RuntimeError(f"{e} — no {split!r} qrels in {repo}") from e
    out = _convert_native(native_corpus, native_queries, native_qrels,
                          repo, "mteb", split, max_queries, dest, urls,
                          os.path.join(dest, "corpus.jsonl"))
    shutil.rmtree(tmp, ignore_errors=True)
    return out


def _pull_beir(repo: str, key: str, dest: str, split: str,
               max_queries: int | None, fetcher) -> dict:
    siblings = hf_siblings(repo)
    corpus_shards = sorted(s for s in siblings
                           if s.startswith("corpus/") and s.endswith(".parquet"))
    query_shards = sorted(s for s in siblings
                          if s.startswith("queries/") and s.endswith(".parquet"))
    if not corpus_shards or not query_shards:
        raise RuntimeError(f"{repo}: no corpus/queries parquet found")
    try:
        qrels_repo = REGISTRY[key]["beir_qrels"]
    except KeyError:
        raise RuntimeError(f"{key}: no BeIR qrels repo known; use --source mteb")
    tmp = os.path.join(dest, ".native")
    os.makedirs(tmp, exist_ok=True)
    corpus_rows: list[dict] = []
    for shard in corpus_shards:
        local = os.path.join(tmp, os.path.basename(shard))
        print(f"downloading {shard} ({repo}) ...")
        fetcher(HF_RESOLVE.format(repo=repo, path=shard), local)
        corpus_rows.extend(read_parquet_rows(local))
        os.remove(local)
    query_rows: list[dict] = []
    for shard in query_shards:
        local = os.path.join(tmp, os.path.basename(shard))
        print(f"downloading {shard} ({repo}) ...")
        fetcher(HF_RESOLVE.format(repo=repo, path=shard), local)
        query_rows.extend(read_parquet_rows(local))
        os.remove(local)
    qrels_name = "test.tsv" if split == "test" else f"{split}.tsv"
    qrels_url = HF_RESOLVE.format(repo=qrels_repo, path=qrels_name)
    native_qrels = os.path.join(tmp, "qrels.tsv")
    print(f"downloading {qrels_name} ({qrels_repo}) ...")
    try:
        fetcher(qrels_url, native_qrels)
    except RuntimeError as e:
        raise RuntimeError(f"{e} — no {split!r} qrels in {qrels_repo}") from e
    urls = {"corpus": HF_RESOLVE.format(repo=repo, path=corpus_shards[0]),
            "queries": HF_RESOLVE.format(repo=repo, path=query_shards[0]),
            "qrels": qrels_url}
    out = _convert_parquet_rows(corpus_rows, query_rows, native_qrels,
                                repo, split, max_queries, dest, urls)
    shutil.rmtree(tmp, ignore_errors=True)
    return out


def _convert_native(native_corpus: str, native_queries: str, native_qrels: str,
                    repo: str, source: str, split: str,
                    max_queries: int | None, dest: str, urls: dict,
                    corpus_path: str) -> dict:
    """Stream-convert native BEIR jsonl files to procure format on disk."""
    n_docs = 0
    with open(native_corpus, encoding="utf-8") as fin, \
            open(corpus_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = _convert_corpus_row(json.loads(line))
            if row is None:
                continue
            fout.write(json.dumps(row) + "\n")
            n_docs += 1
    if n_docs == 0:
        raise RuntimeError(f"{repo}: corpus converted to zero documents")
    qtext: dict[str, str] = {}
    with open(native_queries, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = str(row.get("_id") or "").strip()
            if qid and str(row.get("text") or "").strip():
                qtext[qid] = str(row["text"])
    return _write_queries(qtext, native_qrels, repo, source, split,
                          max_queries, dest, urls, n_docs, corpus_path)


def _convert_parquet_rows(corpus_rows: list[dict], query_rows: list[dict],
                          native_qrels: str, repo: str, split: str,
                          max_queries: int | None, dest: str,
                          urls: dict) -> dict:
    corpus_path = os.path.join(dest, "corpus.jsonl")
    n_docs = 0
    with open(corpus_path, "w", encoding="utf-8") as fout:
        for row in corpus_rows:
            conv = _convert_corpus_row(row)
            if conv is None:
                continue
            fout.write(json.dumps(conv) + "\n")
            n_docs += 1
    if n_docs == 0:
        raise RuntimeError(f"{repo}: corpus converted to zero documents")
    qtext = {str(r.get("_id") or "").strip(): str(r.get("text") or "")
             for r in query_rows
             if str(r.get("_id") or "").strip()
             and str(r.get("text") or "").strip()}
    return _write_queries(qtext, native_qrels, repo, "beir", split,
                          max_queries, dest, urls, n_docs, corpus_path)


def _write_queries(qtext: dict[str, str], native_qrels: str, repo: str,
                   source: str, split: str, max_queries: int | None,
                   dest: str, urls: dict, n_docs: int, corpus_path: str) -> dict:
    rel = _read_qrels_tsv(native_qrels)
    judged = [(qid, cids) for qid, cids in rel.items() if qid in qtext]
    if max_queries is not None:
        judged = judged[:max(0, max_queries)]
    if not judged:
        raise RuntimeError(f"{repo}: no {split!r} queries survived the qrels join")
    queries_path = os.path.join(dest, "queries.jsonl")
    with open(queries_path, "w", encoding="utf-8") as f:
        for qid, cids in judged:
            f.write(json.dumps({
                "query": qtext[qid], "relevant_doc_ids": sorted(cids),
                "relevant_chunk_ids": []}) + "\n")
    kept_qrels = os.path.join(dest, "qrels.tsv")
    with open(native_qrels, "rb") as fin, open(kept_qrels, "wb") as fout:
        fout.write(fin.read())
    manifest = {
        "dataset": repo, "source": source, "split": split,
        "urls": urls, "num_docs": n_docs, "num_queries": len(judged),
        "max_queries": max_queries,
        "pulled_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    manifest_path = os.path.join(dest, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return {"corpus": corpus_path, "queries": queries_path,
            "qrels": kept_qrels, "manifest": manifest_path,
            "num_docs": n_docs, "num_queries": len(judged)}
