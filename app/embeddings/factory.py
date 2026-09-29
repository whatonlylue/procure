"""Embedder factory: env flag selects backend, default stays local."""
from __future__ import annotations

from app.config import Settings
from app.embeddings.base import Embedder


def get_embedder(settings: Settings) -> Embedder:
    if settings.embeddings == "openai":
        from app.embeddings.cloud import OpenAIEmbedder

        return OpenAIEmbedder(settings.openai_api_key, settings.openai_model, settings.openai_base)
    if settings.embeddings == "sbert":
        from app.embeddings.sbert import SbertEmbedder

        return SbertEmbedder(settings.local_model)
    from app.embeddings.local_hash import HashEmbedder

    return HashEmbedder(settings.hash_dim)
