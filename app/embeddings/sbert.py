"""Local neural embedder (sentence-transformers). Lazy import; needs [sbert] extra."""
from __future__ import annotations


class SbertEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError(
                "sentence-transformers not installed. Run: uv pip install -e '.[sbert]'"
            ) from e
        self.model = SentenceTransformer(model_name)
        self.dim: int = int(self.model.get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()
