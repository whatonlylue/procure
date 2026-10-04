"""Local embedding backend: granite-embedding-small-english-r2 via ONNX Runtime.

Single-file replacement for the old hashed bag-of-words embedder. The model
(47M params, 384 dims, up to 8192 tokens, Apache-2.0) runs locally through
ONNX Runtime, which dispatches to whatever accelerator the installed wheel
exposes: CUDA on NVIDIA boxes, CoreML on Apple Silicon, DirectML on Windows,
ROCm/MIGraphX on AMD, OpenVINO on Intel, CPU everywhere else.

Runtime deps (see pyproject): ``onnxruntime``, ``tokenizers``, and
``huggingface_hub`` (model download + tokenizer only -- no torch,
no transformers, no sentence-transformers at runtime).

Tuning (env flags, also mirrored on
:class:`app.config.Settings`):

- ``PROCURE_EMBEDDING_MODEL`` -- HF repo (default
  ``onnx-community/granite-embedding-small-english-r2-ONNX``).
- ``PROCURE_EMBEDDING_FILE`` -- model file inside the repo (default
  ``onnx/model_fp16.onnx``). The fp16 graph is the default: identical
  retrieval quality to fp32 in practice, roughly half the memory, and on
  Apple Silicon it avoids a multi-GB per-shape compile cache that the
  fp32 graph triggers under CoreML. The repo also ships ``onnx/model.onnx``
  (fp32) and the smaller ``onnx/model_quantized.onnx`` / ``onnx/model_q4*.onnx``
  variants, which are CPU-lean and may silently fall back to CPU-only
  kernels on some GPUs.
- ``PROCURE_EMBEDDING_DIR`` -- use a pre-downloaded repo checkout instead
  of the HF cache (offline use).
- ``PROCURE_EMBEDDING_PROVIDERS`` -- comma-separated override, e.g.
  ``cpu`` to pin CPU or ``cuda,coreml,dml,rocm,migraphx,openvino,cpu``.
  Short aliases (``dml``) and canonical names (``DmlExecutionProvider``)
  both work. ``tensorrt`` is honored only when explicitly requested:
  TensorRT builds its engine on first run, which can take minutes.
- ``PROCURE_EMBEDDING_BATCH`` -- texts per inference step (default 32).
- ``PROCURE_EMBEDDING_MAX_LENGTH`` -- token cap per text (default 2048,
  hard-capped at the model's 8192).
"""
from __future__ import annotations

import os
import re
import shutil
import threading

import numpy as np

MODEL_ID = "onnx-community/granite-embedding-small-english-r2-ONNX"
MODEL_FILE = "onnx/model_fp16.onnx"
DIM = 384
MAX_CONTEXT = 8192

#: Stable id recorded next to stored vectors so a backend switch (or the
#: legacy hash corpus, which has no tag) is reported instead of silently
#: serving nonsense from dimension-compatible but meaningless vectors.
BACKEND = "granite-r2"

#: Preferred execution providers, fastest-first. Intersected with what the
#: installed onnxruntime wheel actually exposes; CPU is always last.
#: TensorRT is deliberately absent: it reports as available without its
#: system libraries installed and compiles an engine on first run that can
#: take minutes, so it is only used when explicitly requested.
_PREFERRED_PROVIDERS = (
    "CUDAExecutionProvider",
    "CoreMLExecutionProvider",
    "ROCMExecutionProvider",
    "MIGraphXExecutionProvider",
    "DmlExecutionProvider",
    "OpenVINOExecutionProvider",
    "CPUExecutionProvider",
)

_ALIASES = {
    "cuda": "CUDAExecutionProvider",
    "coreml": "CoreMLExecutionProvider",
    "rocm": "ROCMExecutionProvider",
    "migraphx": "MIGraphXExecutionProvider",
    "dml": "DmlExecutionProvider",
    "directml": "DmlExecutionProvider",
    "openvino": "OpenVINOExecutionProvider",
    "cpu": "CPUExecutionProvider",
    "tensorrt": "TensorRTExecutionProvider",
}


def parse_provider_names(value: str | object | None) -> tuple[str, ...]:
    """Normalize a providers override to canonical execution-provider names.

    Accepts a comma-separated string (``"cuda,cpu"``), a sequence of names,
    or None. Short aliases and canonical names both work; unknown names are
    dropped. Pure function so provider selection is unit-testable.
    """
    if value is None:
        return ()
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",")]
    else:
        parts = [str(p).strip() for p in value]  # type: ignore[union-attr]
    out: list[str] = []
    for part in parts:
        if not part:
            continue
        canon = _ALIASES.get(part.lower(), part)
        # Accept any *ExecutionProvider spelling, even ones this ORT build
        # does not know (filtered against availability later).
        if canon not in out and canon.endswith("ExecutionProvider"):
            out.append(canon)
    return tuple(out)


def resolve_providers(available: list[str] | tuple[str, ...],
                      requested: tuple[str, ...] = ()) -> list[str]:
    """Pick the execution-provider order for an InferenceSession.

    Explicitly requested providers that exist come first (in request
    order), then the built-in fastest-first preference, filtered to what
    is actually installed. CPU is always last when available so a GPU
    failure still degrades to working inference instead of an exception.
    Pure function so the selection policy is unit-testable.
    """
    avail = list(available) or ["CPUExecutionProvider"]
    picked = [p for p in requested if p in avail]
    for provider in _PREFERRED_PROVIDERS:
        if provider in avail and provider not in picked:
            picked.append(provider)
    if not picked:
        return avail  # unknown-only runtime: trust its own order
    if "CPUExecutionProvider" in avail and "CPUExecutionProvider" not in picked:
        picked.append("CPUExecutionProvider")
    return picked


class GraniteEmbedder:
    """Granite R2 embeddings over ONNX Runtime, CLS-pooled + L2-normalized.

    The model card (and its sentence-transformers wiring) defines the
    pooling: first-token (CLS) slice of the last hidden state, then L2
    normalize for cosine search. Model download + session creation are
    lazy (first :meth:`embed` call) and thread-safe, so importing this
    module and constructing the embedder never touch the network.
    """

    dim = DIM
    backend = BACKEND

    def __init__(self, model: str = MODEL_ID, model_file: str = MODEL_FILE,
                 model_dir: str | None = None,
                 providers: str | list[str] | tuple[str, ...] | None = None,
                 batch_size: int = 32, max_length: int = 2048) -> None:
        self.model = model or MODEL_ID
        self.model_file = model_file or MODEL_FILE
        self.model_dir = model_dir or None
        self.requested_providers = parse_provider_names(providers)
        self.batch_size = max(1, int(batch_size))
        self.max_length = max(1, min(int(max_length), MAX_CONTEXT))
        self._lock = threading.Lock()
        self._session = None  # ort.InferenceSession, created lazily
        self._tokenizer = None  # tokenizers.Tokenizer, created lazily
        self._input_names: tuple[str, ...] = ()
        self._active_providers: tuple[str, ...] = ()

    # -- introspection (never loads the model) --------------------------
    @property
    def loaded(self) -> bool:
        return self._session is not None

    @property
    def provider(self) -> str | None:
        """Highest-priority provider the session actually runs on."""
        return self._active_providers[0] if self._active_providers else None

    def info(self) -> dict:
        """Cheap status for /api/health: no download, no inference."""
        return {
            "backend": self.backend,
            "model": self.model,
            "file": self.model_file,
            "dim": self.dim,
            "providers_requested": list(self.requested_providers),
            "providers_active": list(self._active_providers),
            "loaded": self.loaded,
        }

    # -- public API ------------------------------------------------------
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts to L2-normalized 384-dim vectors."""
        if not texts:
            return []
        self._ensure_loaded()
        assert self._session is not None and self._tokenizer is not None
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            chunk = [t if isinstance(t, str) else "" for t in texts[i:i + self.batch_size]]
            out.extend(self._encode_batch(chunk))
        return out

    # -- lazy setup ------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._session is not None:
            return
        with self._lock:
            if self._session is not None:
                return
            try:
                import onnxruntime as ort
            except ImportError as e:
                raise RuntimeError(
                    "The embedding backend needs the 'onnxruntime' package "
                    "(pip install onnxruntime; NVIDIA GPUs: onnxruntime-gpu). "
                    f"Requested providers: {list(self.requested_providers) or 'auto'}."
                ) from e
            model_path, tokenizer_path = self._resolve_files()
            tokenizer = self._load_tokenizer(tokenizer_path)
            providers = resolve_providers(
                list(ort.get_available_providers()), self.requested_providers)
            opts = ort.SessionOptions()
            opts.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL)
            opts.log_severity_level = 3
            session = ort.InferenceSession(str(model_path), sess_options=opts,
                                           providers=providers)
            self._input_names = tuple(i.name for i in session.get_inputs())
            self._active_providers = tuple(session.get_providers())
            self._session = session
            self._tokenizer = tokenizer
            # Warmup: compiles/pins the GPU path (CoreML, CUDA graphs) and
            # validates the output width before any document is ingested.
            width = self._encode_batch(["procure warmup"])
            if len(width[0]) != self.dim:
                raise RuntimeError(
                    f"Embedding model output width {len(width[0])} != "
                    f"expected {self.dim} ({self.model}:{self.model_file}).")

    def _resolve_files(self) -> tuple[str, str]:
        """Return (model_path, tokenizer_path), downloading once if needed."""
        if self.model_dir is not None:
            root = os.path.abspath(os.path.expanduser(self.model_dir))
            model_path = os.path.join(root, self.model_file)
            tokenizer_path = os.path.join(root, "tokenizer.json")
            missing = [p for p in (model_path, tokenizer_path)
                       if not os.path.isfile(p)]
            if missing:
                raise RuntimeError(
                    f"PROCURE_EMBEDDING_DIR={root} is missing "
                    f"{', '.join(missing)}; expected a checkout of "
                    f"{self.model} (or unset it to download).")
            # A real checkout holds real files, so ONNX external-data
            # references resolve; only the HF blob cache needs flattening.
            if not os.path.islink(model_path):
                return model_path, tokenizer_path
            return self._materialize(root, os.path.basename(root)), tokenizer_path
        try:
            from huggingface_hub import snapshot_download
        except ImportError as e:
            raise RuntimeError(
                "The embedding backend needs 'huggingface_hub' to fetch "
                f"{self.model} (or set PROCURE_EMBEDDING_DIR to a "
                "pre-downloaded checkout).") from e
        data_file = self.model_file + "_data"
        try:
            root = snapshot_download(
                repo_id=self.model,
                allow_patterns=[self.model_file, data_file, "tokenizer.json",
                                "tokenizer_config.json", "config.json",
                                "special_tokens_map.json"],
            )
        except Exception as e:
            raise RuntimeError(
                f"Could not download embedding model {self.model} "
                f"({self.model_file}): {e}. Pass a local checkout via "
                "PROCURE_EMBEDDING_DIR for offline use.") from e
        return (self._materialize(root, os.path.basename(root.rstrip(os.sep))),
                os.path.join(root, "tokenizer.json"))

    def _materialize(self, root: str, commit: str) -> str:
        """Flatten symlinked model files into real files for ONNX Runtime.

        The Hugging Face cache stores blobs under hashed names and exposes
        them via symlinks, but ONNX external-data references (``*.onnx_data``)
        are validated against the model's real directory, so loading through
        the symlink farm fails. This copies (hardlinks when possible) the
        model file plus its ``*_data`` sidecar into a per-model cache dir
        laid out exactly like the repo, once per commit.
        """
        from platformdirs import user_cache_dir

        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", self.model)
        base = (os.environ.get("PROCURE_EMBEDDING_CACHE_DIR")
                or os.path.join(user_cache_dir("procure"), "models"))
        dest = os.path.join(base, slug)
        marker = os.path.join(dest, ".commit")
        try:
            with open(marker, encoding="utf-8") as f:
                current = f.read().strip()
        except OSError:
            current = None
        wanted = [self.model_file]
        sidecar = self.model_file + "_data"
        if os.path.isfile(os.path.join(root, sidecar)):
            wanted.append(sidecar)
        dest_model = os.path.join(dest, self.model_file)
        if current != commit or not all(
                os.path.isfile(os.path.join(dest, w)) for w in wanted):
            for rel in wanted:
                self._link_or_copy(os.path.join(root, rel),
                                   os.path.join(dest, rel))
            os.makedirs(dest, exist_ok=True)
            tmp = marker + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(commit)
            os.replace(tmp, marker)
        return dest_model

    @staticmethod
    def _link_or_copy(src: str, dst: str) -> None:
        try:
            if (os.path.isfile(dst)
                    and os.path.getsize(dst) == os.path.getsize(src)):
                return
        except OSError:
            pass
        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        tmp = dst + ".tmp"
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
            try:
                os.link(src, tmp)  # same filesystem: no extra disk used
            except OSError:
                shutil.copyfile(src, tmp)
            os.replace(tmp, dst)
        finally:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except OSError:
                pass

    def _load_tokenizer(self, tokenizer_path: str):
        try:
            from tokenizers import Tokenizer
        except ImportError as e:
            raise RuntimeError(
                "The embedding backend needs the 'tokenizers' package.") from e
        path = str(tokenizer_path)
        if not os.path.isfile(path):
            raise RuntimeError(
                f"Tokenizer file missing: {path} (repo {self.model}).")
        tokenizer = Tokenizer.from_file(path)
        pad_id = tokenizer.token_to_id("[PAD]")
        if pad_id is None:
            pad_id = 0
        tokenizer.enable_truncation(max_length=self.max_length)
        tokenizer.enable_padding(pad_id=pad_id, pad_token="[PAD]")
        return tokenizer

    # -- inference -------------------------------------------------------
    def _encode_batch(self, texts: list[str]) -> list[list[float]]:
        assert self._session is not None and self._tokenizer is not None
        encodings = self._tokenizer.encode_batch(texts)
        ids = np.asarray([e.ids for e in encodings], dtype=np.int64)
        mask = np.asarray([e.attention_mask for e in encodings], dtype=np.int64)
        feed: dict[str, np.ndarray] = {}
        if "input_ids" in self._input_names:
            feed["input_ids"] = ids
        if "attention_mask" in self._input_names:
            feed["attention_mask"] = mask
        if "token_type_ids" in self._input_names:
            feed["token_type_ids"] = np.zeros_like(ids)
        if not feed:
            raise RuntimeError(
                f"ONNX graph expects {list(self._input_names)}; this "
                "embedder only feeds input_ids/attention_mask/token_type_ids.")
        hidden = self._session.run(None, feed)[0]
        cls = np.asarray(hidden[:, 0, :], dtype=np.float32)
        norms = np.linalg.norm(cls, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (cls / norms).astype(np.float32).tolist()
