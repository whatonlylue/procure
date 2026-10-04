"""Chroma vector store. Lazy import; needs [chroma] extra. Opt-in via env."""
from __future__ import annotations

from typing import Any

from app.vectordb.base import ChunkHit, chunk_index

_PAGE = 5000


class ChromaVectorStore:
    def __init__(self, persist_dir: str):
        try:
            import chromadb
        except ImportError as e:
            raise RuntimeError(
                "chromadb not installed. Run: uv pip install -e '.[chroma]'"
            ) from e
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            "chunks", metadata={"hnsw:space": "cosine"}
        )

    def upsert(self, ids: list[str], doc_ids: list[str],
               embeddings: list[list[float]], texts: list[str]) -> None:
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=[{"doc_id": d} for d in doc_ids],
        )

    def _paged_get(self, where: dict[str, Any] | None,
                   include: list[str]) -> tuple[list[str], list[str], list[dict]]:
        """Fetch every matching row, paging past server-side limits."""
        ids: list[str] = []
        docs: list[str] = []
        metas: list[dict] = []
        offset = 0
        while True:
            got = self.collection.get(where=where, limit=_PAGE, offset=offset,
                                      include=include)
            page_ids = list(got.get("ids", []) or [])
            if not page_ids:
                break
            ids.extend(page_ids)
            if "documents" in include:
                docs.extend(list(got.get("documents", []) or []))
            if "metadatas" in include:
                metas.extend(list(got.get("metadatas", []) or []))
            if len(page_ids) < _PAGE:
                break
            offset += _PAGE
        return ids, docs, metas

    def delete_by_doc(self, doc_id: str) -> int:
        ids, _, _ = self._paged_get({"doc_id": doc_id}, include=[])
        for i in range(0, len(ids), _PAGE):
            self.collection.delete(ids=ids[i:i + _PAGE])
        return len(ids)

    def search(self, query_embedding: list[float], top_k: int,
               doc_ids: list[str] | None = None) -> list[ChunkHit]:
        where = {"doc_id": {"$in": doc_ids}} if doc_ids else None
        res = self.collection.query(
            query_embeddings=[query_embedding], n_results=top_k, where=where,
            include=["documents", "metadatas", "distances"],
        )
        ids = res.get("ids", [[]])[0] or []
        dists = (res.get("distances") or [[]])[0] or []
        metas = (res.get("metadatas") or [[]])[0] or []
        docs = (res.get("documents") or [[]])[0] or []
        hits: list[ChunkHit] = []
        for i, cid in enumerate(ids):
            dist = dists[i] if i < len(dists) else 0.0
            meta = metas[i] if i < len(metas) else {}
            text = docs[i] if i < len(docs) else ""
            hits.append(ChunkHit(
                chunk_id=cid,
                doc_id=(meta or {}).get("doc_id", ""),
                text=text or "",
                score=float(1.0 - (dist or 0.0)),
            ))
        return hits

    def get_by_doc(self, doc_id: str) -> list[tuple[str, str]]:
        """(chunk_id, text) rows for one document, in chunk order."""
        ids, docs, _ = self._paged_get({"doc_id": doc_id},
                                       include=["documents"])
        rows = list(zip(ids, [d or "" for d in docs]))
        return sorted(rows, key=lambda r: chunk_index(r[0]))

    def count(self) -> int:
        return int(self.collection.count())

    def all_chunks(self) -> list[tuple[str, str, str]]:
        """All (chunk_id, doc_id, text) rows; used to backfill sparse stats."""
        ids, docs, metas = self._paged_get(
            None, include=["documents", "metadatas"])
        return [(cid, (m or {}).get("doc_id", ""), d or "")
                for cid, m, d in zip(ids, metas, docs)]
