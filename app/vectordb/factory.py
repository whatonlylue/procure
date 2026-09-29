"""VectorStore factory: env flag selects backend, default stays local."""
from __future__ import annotations

from app.config import Settings
from app.vectordb.base import VectorStore


def get_vector_store(settings: Settings) -> VectorStore:
    if settings.vectordb == "chroma":
        from app.vectordb.chroma_vec import ChromaVectorStore

        return ChromaVectorStore(settings.chroma_dir)
    from app.vectordb.sqlite_vec import SqliteVectorStore

    return SqliteVectorStore(settings.vec_db)
