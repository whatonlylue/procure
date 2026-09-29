"""Cloud embedder (OpenAI-compatible API). Stdlib HTTP only. Opt-in via env."""
from __future__ import annotations

import json
import urllib.request


class OpenAIEmbedder:
    def __init__(self, api_key: str, model: str, base_url: str):
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for PROCURE_EMBEDDINGS=openai")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.dim: int = len(self.embed(["dimension probe"])[0])

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), 100):
            batch = texts[i : i + 100]
            req = urllib.request.Request(
                f"{self.base_url}/embeddings",
                data=json.dumps({"model": self.model, "input": batch}).encode(),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                payload = json.load(resp)
            out.extend([d["embedding"] for d in payload["data"]])
        return out
