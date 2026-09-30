"""Central settings. Local-first defaults; cloud via env flags only."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _default_data_dir() -> str:
    override = os.environ.get("PROCURE_DATA_DIR")
    if override:
        return override
    from platformdirs import user_data_dir

    return user_data_dir("procure")


def _truthy(value: str) -> bool:
    return value.strip().lower() not in ("", "0", "false", "no", "off")


@dataclass
class Settings:
    """Mutable so the workspace can change retrieval settings at runtime
    (see RAGService.update_settings); env flags still set the startup values."""

    data_dir: str = field(default_factory=_default_data_dir)
    # Embeddings backend: "hash" (local default, zero deps) | "sbert" (local neural) | "openai" (cloud)
    embeddings: str = field(default_factory=lambda: _env("PROCURE_EMBEDDINGS", "hash").lower())
    # Vector store: "sqlite" (local default) | "chroma" (local persistent, needs extra)
    vectordb: str = field(default_factory=lambda: _env("PROCURE_VECTORDB", "sqlite").lower())
    local_model: str = field(default_factory=lambda: _env("PROCURE_LOCAL_MODEL", "all-MiniLM-L6-v2"))
    hash_dim: int = field(default_factory=lambda: int(_env("PROCURE_HASH_DIM", "384")))
    openai_api_key: str = field(default_factory=lambda: _env("OPENAI_API_KEY", ""))
    openai_model: str = field(default_factory=lambda: _env("OPENAI_EMBED_MODEL", "text-embedding-3-small"))
    openai_base: str = field(default_factory=lambda: _env("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    chunk_size: int = field(default_factory=lambda: int(_env("PROCURE_CHUNK_SIZE", "1000")))
    chunk_overlap: int = field(default_factory=lambda: int(_env("PROCURE_CHUNK_OVERLAP", "150")))
    # Search: "hybrid" (BM25 + dense with RRF fusion) | "dense" (cosine only)
    search_mode: str = field(default_factory=lambda: _env("PROCURE_SEARCH", "hybrid").lower())
    # Answerability rerank on top of retrieval: "on" | "off"
    rerank: str = field(default_factory=lambda: _env("PROCURE_RERANK", "on").lower())
    top_k: int = field(default_factory=lambda: int(_env("PROCURE_TOPK", "5")))
    # Local cross-encoder model for the answerability reranker (cached only
    # unless PROCURE_CROSS_ENCODER_ALLOW_DOWNLOAD=1); "" disables it.
    cross_encoder_model: str = field(default_factory=lambda: _env(
        "PROCURE_CROSS_ENCODER", "cross-encoder/ms-marco-MiniLM-L6-v2"))
    cross_encoder_download: bool = field(default_factory=lambda: _truthy(_env(
        "PROCURE_CROSS_ENCODER_ALLOW_DOWNLOAD", "0")))
    # Frozen NLI teacher for CLEAR-style answerability (same DeBERTa-v3 NLI
    # family as the paper's teacher); "" disables it.
    nli_model: str = field(default_factory=lambda: _env(
        "PROCURE_NLI_MODEL", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"))
    nli_download: bool = field(default_factory=lambda: _truthy(_env(
        "PROCURE_NLI_ALLOW_DOWNLOAD", "0")))
    # CLEAR inference weight: score = sigmoid(relevance) + alpha * entailment.
    alpha_nli: float = field(default_factory=lambda: float(_env("PROCURE_ALPHA_NLI", "0.5")))
    # MCP server (Streamable HTTP) bind address. Served at http://host:port/mcp.
    mcp_host: str = field(default_factory=lambda: _env("PROCURE_MCP_HOST", "127.0.0.1"))
    mcp_port: int = field(default_factory=lambda: int(_env("PROCURE_MCP_PORT", "8001")))
    # Folder-watch poll interval in seconds (0 disables the background loop;
    # manual "sync now" still works).
    watch_interval: int = field(
        default_factory=lambda: int(_env("PROCURE_WATCH_INTERVAL", "60")))
    # OCR: "auto" (thin/scanned pages only) | "on" (force OCR on every PDF
    # page; images are always OCRed) | "off" (never OCR; scanned files fail
    # with guidance). Needs the [ocr] extra either way.
    ocr_mode: str = field(default_factory=lambda: _env("PROCURE_OCR", "auto").lower())
    # URL fetch timeout in seconds.
    fetch_timeout: float = field(
        default_factory=lambda: float(_env("PROCURE_FETCH_TIMEOUT", "20")))
    # Reject URL fetches that resolve to loopback/link-local/private IPs
    # (SSRF guard for the MCP fetch tool). Set 1 to allow (tests, intranets).
    fetch_allow_private: bool = field(default_factory=lambda: _truthy(_env(
        "PROCURE_FETCH_ALLOW_PRIVATE", "0")))
    # Per-file upload cap in megabytes (413 beyond it).
    max_upload_mb: int = field(
        default_factory=lambda: int(_env("PROCURE_MAX_UPLOAD_MB", "100")))
    # Start the MCP server automatically with the workspace (persisted by
    # the UI toggle; not an env flag).
    mcp_autostart: bool = False

    def __post_init__(self) -> None:
        # Absolute once, so the MCP subprocess can run with any cwd and the
        # frozen sidecar never resolves paths against the bundle location.
        self.data_dir = os.path.abspath(os.path.expanduser(self.data_dir))
        # Runtime overrides persisted by update_settings win over env flags,
        # so a change made in the UI (and read back by a restarted MCP
        # subprocess) survives restarts.
        self._load_overrides()

    def _load_overrides(self) -> None:
        path = os.path.join(self.data_dir, "settings.json")
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return
        if not isinstance(data, dict):
            return
        search = data.get("search")
        if isinstance(search, str) and search.strip().lower() in ("hybrid", "dense"):
            self.search_mode = search.strip().lower()
        rerank = data.get("rerank")
        if isinstance(rerank, str) and rerank.strip().lower() in ("on", "off"):
            self.rerank = rerank.strip().lower()
        if isinstance(data.get("mcp_autostart"), bool):
            self.mcp_autostart = data["mcp_autostart"]

    def save_overrides(self) -> None:
        """Persist runtime-toggled settings (search/rerank/autostart)."""
        os.makedirs(self.data_dir, exist_ok=True)
        path = os.path.join(self.data_dir, "settings.json")
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({
                "search": self.search_mode,
                "rerank": self.rerank,
                "mcp_autostart": self.mcp_autostart,
            }, f, indent=2)
        os.replace(tmp, path)

    @property
    def hybrid(self) -> bool:
        return self.search_mode != "dense"

    @property
    def rerank_enabled(self) -> bool:
        return _truthy(self.rerank)

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.data_dir, "raw")

    @property
    def meta_db(self) -> str:
        return os.path.join(self.data_dir, "meta.sqlite")

    @property
    def vec_db(self) -> str:
        return os.path.join(self.data_dir, "vectordb.sqlite")

    @property
    def chroma_dir(self) -> str:
        return os.path.join(self.data_dir, "chroma")

    @property
    def sparse_db(self) -> str:
        return os.path.join(self.data_dir, "sparse.sqlite")


def get_settings() -> Settings:
    return Settings()
