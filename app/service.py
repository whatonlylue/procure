"""Ingest pipeline + search orchestration."""
from __future__ import annotations

import os
import uuid
from functools import lru_cache

from app import chunking, extract
from app.config import Settings, get_settings
from app.embeddings.factory import get_embedder
from app.search import (
    CrossEncoderScorer,
    NLIEntailmentScorer,
    clear_answerability,
    combine_answerability,
    content_terms,
    coverage_score,
    minmax_norm,
    rrf_fuse,
)
from app.search import SparseStore
from app.store import MetaStore
from app.vectordb.factory import get_vector_store

# Candidate pool sizes for hybrid retrieval and reranking.
_RETRIEVE_MULT = 5
_RETRIEVE_MIN = 25
_RERANK_MULT = 3
_RERANK_MIN = 15
# Final blend: normalized fused retrieval score vs answerability.
_W_FUSED = 0.5
_W_ANSWER = 0.5


class DimensionMismatchError(Exception):
    pass


class RAGService:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        os.makedirs(self.settings.raw_dir, exist_ok=True)
        self.meta = MetaStore(self.settings.meta_db)
        self.embedder = get_embedder(self.settings)
        self.vectors = get_vector_store(self.settings)
        self.sparse = SparseStore(self.settings.sparse_db)
        self._cross = CrossEncoderScorer(
            self.settings.cross_encoder_model,
            self.settings.cross_encoder_download,
        )
        self._nli = NLIEntailmentScorer(
            self.settings.nli_model,
            self.settings.nli_download,
        )
        # One-time backfill for corpora ingested before sparse stats existed.
        if self.sparse.count() == 0 and self.vectors.count() > 0:
            self._backfill_sparse()

    def _backfill_sparse(self) -> None:
        all_chunks = getattr(self.vectors, "all_chunks", None)
        if not callable(all_chunks):
            return
        rows = all_chunks()
        if rows:
            ids = [r[0] for r in rows]
            self.sparse.upsert_chunks(ids, [r[1] for r in rows], [r[2] for r in rows])

    # -- ingest ---------------------------------------------------------
    def ingest_files(self, files: list[tuple[str, bytes]]) -> list[dict]:
        return [self._ingest_one(filename, data) for filename, data in files]

    def _ingest_one(self, filename: str, data: bytes) -> dict:
        doc_id = uuid.uuid4().hex[:12]
        safe = "".join(c if c.isalnum() or c in "._- " else "_" for c in filename).strip() or "file"
        raw_path = os.path.join(self.settings.raw_dir, f"{doc_id}_{safe}")
        self.meta.upsert(doc_id, filename, "processing")
        try:
            if not extract.is_supported(filename):
                raise ValueError(f"Unsupported file type: {filename}")
            with open(raw_path, "wb") as f:
                f.write(data)
            text = extract.extract_text(raw_path, filename)
            return self._commit_text(doc_id, filename, text)
        except Exception as e:  # noqa: BLE001 - per-file failure must not break batch
            self.meta.upsert(doc_id, filename, "failed", 0, str(e))
            return {"doc_id": doc_id, "filename": filename, "status": "failed", "error": str(e)}

    def ingest_text(self, filename: str, text: str) -> dict:
        """Ingest raw text (agent transcripts, tool outputs, notes) as a document.

        Same chunk + embed + store pipeline as file upload; the text is also
        saved under raw_dir so the document stays re-ingestible from Library.
        """
        doc_id = uuid.uuid4().hex[:12]
        safe = "".join(c if c.isalnum() or c in "._- " else "_" for c in filename).strip() or "snippet"
        if os.path.splitext(safe)[1].lower() not in (".txt", ".md", ".markdown"):
            safe += ".md"
        filename = filename.strip() or safe
        raw_path = os.path.join(self.settings.raw_dir, f"{doc_id}_{safe}")
        self.meta.upsert(doc_id, filename, "processing")
        try:
            with open(raw_path, "w", encoding="utf-8") as f:
                f.write(text or "")
            return self._commit_text(doc_id, filename, text)
        except Exception as e:  # noqa: BLE001 - surface as failed status
            self.meta.upsert(doc_id, filename, "failed", 0, str(e))
            return {"doc_id": doc_id, "filename": filename,
                    "status": "failed", "error": str(e)}

    def _commit_text(self, doc_id: str, filename: str, text: str) -> dict:
        chunks = chunking.split_text(
            text, self.settings.chunk_size, self.settings.chunk_overlap
        )
        if not chunks:
            raise ValueError("No text extracted")
        embs = self.embedder.embed(chunks)
        ids = [f"{doc_id}:{i}" for i in range(len(chunks))]
        self.vectors.upsert(ids, [doc_id] * len(chunks), embs, chunks)
        self.sparse.upsert_chunks(ids, [doc_id] * len(chunks), chunks)
        self.meta.upsert(doc_id, filename, "ready", len(chunks))
        return {"doc_id": doc_id, "filename": filename, "status": "ready",
                "chunk_count": len(chunks)}

    # -- library --------------------------------------------------------
    def list_documents(self) -> list[dict]:
        return self.meta.list()

    def get_document(self, doc_id: str) -> dict:
        """Full document record: metadata plus complete extracted text.

        The text is re-extracted from the saved raw file when it still
        exists; otherwise the stored chunks are joined in order as a
        fallback. Raises KeyError for unknown doc ids.
        """
        doc = self.meta.get(doc_id)
        if not doc:
            raise KeyError(doc_id)
        text = ""
        raw_path = self._raw_path(doc_id)
        if raw_path is not None:
            try:
                text = extract.extract_text(raw_path, doc["filename"])
            except Exception:  # noqa: BLE001 - fall back to stored chunks
                text = ""
        if not text:
            rows = self.vectors.get_by_doc(doc_id)
            text = "\n\n".join(t for _, t in rows if t)
        return {**doc, "text": text}

    def _raw_path(self, doc_id: str) -> str | None:
        try:
            names = os.listdir(self.settings.raw_dir)
        except OSError:
            return None
        return next(
            (os.path.join(self.settings.raw_dir, n) for n in names
             if n.startswith(doc_id + "_")),
            None,
        )

    # -- runtime settings -----------------------------------------------
    def update_settings(self, search: str | None = None,
                        rerank: str | None = None) -> dict:
        """Change retrieval settings at runtime (workspace process only).

        Raises ValueError on unknown values. The MCP server subprocess
        reads these values when it is (re)started.
        """
        if search is not None:
            mode = search.strip().lower()
            if mode not in ("hybrid", "dense"):
                raise ValueError(f"Unknown search mode: {search!r} (want hybrid|dense)")
            self.settings.search_mode = mode
        if rerank is not None:
            flag = rerank.strip().lower()
            if flag not in ("on", "off"):
                raise ValueError(f"Unknown rerank flag: {rerank!r} (want on|off)")
            self.settings.rerank = flag
        return {"search": self.settings.search_mode, "rerank": self.settings.rerank}

    def delete_document(self, doc_id: str) -> bool:
        doc = self.meta.get(doc_id)
        if not doc:
            return False
        self.vectors.delete_by_doc(doc_id)
        self.sparse.delete_by_doc(doc_id)
        for name in os.listdir(self.settings.raw_dir):
            if name.startswith(doc_id + "_"):
                try:
                    os.remove(os.path.join(self.settings.raw_dir, name))
                except OSError:
                    pass
        self.meta.delete(doc_id)
        return True

    def reingest_document(self, doc_id: str) -> dict:
        doc = self.meta.get(doc_id)
        if not doc:
            raise KeyError(doc_id)
        raw_path = self._raw_path(doc_id)
        if raw_path is None:
            raise FileNotFoundError(f"Raw file for {doc_id} is gone; re-upload it")
        self.vectors.delete_by_doc(doc_id)
        self.sparse.delete_by_doc(doc_id)
        self.meta.upsert(doc_id, doc["filename"], "processing")
        try:
            text = extract.extract_text(raw_path, doc["filename"])
            chunks = chunking.split_text(
                text, self.settings.chunk_size, self.settings.chunk_overlap
            )
            if not chunks:
                raise ValueError("No text extracted")
            embs = self.embedder.embed(chunks)
            ids = [f"{doc_id}:{i}" for i in range(len(chunks))]
            self.vectors.upsert(ids, [doc_id] * len(chunks), embs, chunks)
            self.sparse.upsert_chunks(ids, [doc_id] * len(chunks), chunks)
            self.meta.upsert(doc_id, doc["filename"], "ready", len(chunks))
            return {"doc_id": doc_id, "filename": doc["filename"],
                    "status": "ready", "chunk_count": len(chunks)}
        except Exception as e:  # noqa: BLE001 - surface as failed status
            self.meta.upsert(doc_id, doc["filename"], "failed", 0, str(e))
            return {"doc_id": doc_id, "filename": doc["filename"],
                    "status": "failed", "error": str(e)}

    # -- search ---------------------------------------------------------
    def search(self, query: str, top_k: int = 5,
               doc_ids: list[str] | None = None) -> list[dict]:
        if not query.strip():
            return []
        hybrid = self.settings.hybrid
        rerank_on = self.settings.rerank_enabled
        depth = max(top_k * _RETRIEVE_MULT, _RETRIEVE_MIN)
        q = self.embedder.embed([query])[0]
        dense_hits = self.vectors.search(q, depth if (hybrid or rerank_on) else top_k,
                                         doc_ids)
        dense_by_id = {h.chunk_id: h for h in dense_hits}

        sparse_by_id: dict[str, float] = {}
        sparse_text: dict[str, tuple[str, str]] = {}
        if hybrid:
            for s in self.sparse.search(query, depth, doc_ids):
                sparse_by_id[s.chunk_id] = s.score
                sparse_text[s.chunk_id] = (s.doc_id, s.text)

        if hybrid:
            fused = rrf_fuse(
                [[h.chunk_id for h in dense_hits], list(sparse_by_id.keys())])
            ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
        else:
            fused = {h.chunk_id: h.score for h in dense_hits}
            ordered = [(h.chunk_id, h.score) for h in dense_hits]
        fused_rank = {cid: i + 1 for i, (cid, _) in enumerate(ordered)}

        pool = [cid for cid, _ in ordered[:max(top_k * _RERANK_MULT, _RERANK_MIN)]]

        texts = {
            cid: (dense_by_id[cid].text if cid in dense_by_id
                  else sparse_text[cid][1])
            for cid in pool if cid in dense_by_id or cid in sparse_text
        }
        doc_of = {
            cid: (dense_by_id[cid].doc_id if cid in dense_by_id
                  else sparse_text[cid][0])
            for cid in texts
        }

        answer: dict[str, float] = {}
        entail: dict[str, float] = {}
        if rerank_on and texts:
            pool_texts = [texts[cid] for cid in texts]
            cross = self._cross.score(query, pool_texts)
            ent = self._nli.score(query, pool_texts)
            if cross is not None and ent is not None:
                # CLEAR inference (mode "all"): sigmoid(relevance) + alpha * entailment.
                for cid, a, e in zip(texts, clear_answerability(
                        cross, ent, self.settings.alpha_nli), ent):
                    answer[cid] = a
                    entail[cid] = e
            else:
                idf = self.sparse.idf_map(content_terms(query))
                cov = [coverage_score(query, texts[cid], idf) for cid in texts]
                for cid, a in zip(texts, combine_answerability(cross, cov)):
                    answer[cid] = a

        names = {d["doc_id"]: d["filename"] for d in self.meta.list()}
        if rerank_on and texts:
            fnorm = minmax_norm({cid: fused[cid] for cid in texts})
            final = {cid: _W_FUSED * fnorm[cid] + _W_ANSWER * answer[cid]
                     for cid in texts}
            ranked = sorted(final.items(), key=lambda kv: (-kv[1], fused_rank[kv[0]]))
        else:
            ranked = [(cid, fused[cid]) for cid in pool if cid in texts]

        out = []
        for cid, score in ranked[:top_k]:
            did = doc_of[cid]
            out.append({
                "chunk_id": cid, "doc_id": did,
                "filename": names.get(did, did),
                "text": texts[cid], "score": round(float(score), 4),
                "sparse_score": round(float(sparse_by_id.get(cid, 0.0)), 4),
                "dense_score": round(float(
                    dense_by_id[cid].score if cid in dense_by_id else 0.0), 4),
                "fused_rank": fused_rank[cid],
                "answerability": round(float(answer.get(cid, 0.0)), 4),
                "entailment": round(float(entail.get(cid, 0.0)), 4),
            })
        return out


@lru_cache(maxsize=1)
def get_service() -> RAGService:
    return RAGService()
