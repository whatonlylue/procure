"""Chroma vector store. Lazy import; needs [chroma] extra. Opt-in via env."""
from __future__ import annotations

from app.vectordb.base import ChunkHit


def _chunk_index(chunk_id: str) -> int:
    """Numeric suffix of a `{doc_id}:{i}` chunk id; unknown shapes sort last."""
    try:
        return int(chunk_id.rsplit(":", 1)[1])
    except (IndexError, ValueError):
        return 1 << 30


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

    def delete_by_doc(self, doc_id: str) -> int:
        existing = self.collection.get(where={"doc_id": doc_id}, limit=100000)
        ids = existing.get("ids", [])
        if ids:
            self.collection.delete(ids=ids)
        return len(ids)

    def search(self, query_embedding: list[float], top_k: int,
               doc_ids: list[str] | None = None) -> list[ChunkHit]:
        where = {"doc_id": {"$in": doc_ids}} if doc_ids else None
        res = self.collection.query(
            query_embeddings=[query_embedding], n_results=top_k, where=where,
            include=["documents", "metadatas", "distances"],
        )
        hits: list[ChunkHit] = []
        for i, cid in enumerate(res["ids"][0]):
            dist = (res["distances"][0] or [0.0])[i] if res.get("distances") else 0.0
            hits.append(ChunkHit(
                chunk_id=cid,
                doc_id=(res["metadatas"][0][i] or {}).get("doc_id", ""),
                text=(res["documents"][0] or [""])[i] or "",
                score=float(1.0 - (dist or 0.0)),
            ))
        return hits

    def get_by_doc(self, doc_id: str) -> list[tuple[str, str]]:
        """(chunk_id, text) rows for one document, in chunk order."""
        got = self.collection.get(
            where={"doc_id": doc_id}, limit=100000, include=["documents"])
        rows = list(zip(got.get("ids", []) or [], got.get("documents", []) or []))
        return sorted(rows, key=lambda r: _chunk_index(r[0]))

    def count(self) -> int:
        return int(self.collection.count())

    def all_chunks(self) -> list[tuple[str, str, str]]:
        """All (chunk_id, doc_id, text) rows; used to backfill sparse stats."""
        got = self.collection.get(limit=1000000, include=["documents", "metadatas"])
        ids = got.get("ids", []) or []
        docs = got.get("documents", []) or []
        metas = got.get("metadatas", []) or []
        return [(cid, (m or {}).get("doc_id", ""), d or "")
                for cid, m, d in zip(ids, metas, docs)]
