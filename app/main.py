"""FastAPI app: /api/* + static frontend."""
from __future__ import annotations

import importlib.util
import os
import urllib.parse
from contextlib import asynccontextmanager
from functools import lru_cache

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware

from app import extract, logbuffer
from app.config import get_settings
from app.jobs import get_job_manager
from app.mcp_manager import get_mcp_manager
from app.ocr import backend_name as ocr_backend
from app.service import get_service
from app.version import __version__

# Hosts the workspace server may serve. The UI is same-origin on one of
# these; anything else is a DNS-rebinding read or a cross-site drive-by
# write and is rejected (no auth tokens; this is a localhost-only app).
_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1", "[::1]", "testserver"}

_CSP = (
    "default-src 'self'; "
    "script-src 'self' 'unsafe-inline'; "
    "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
    "font-src 'self' https://fonts.gstatic.com data:; "
    "img-src 'self' data:; "
    "connect-src 'self' http://127.0.0.1:* http://localhost:*"
)


def _host_ok(host: str) -> bool:
    return (host or "").split(":")[0].strip("[]").lower() in _LOCAL_HOSTS


def _origin_ok(value: str | None) -> bool:
    if not value or value == "null":
        return True
    try:
        parts = urllib.parse.urlparse(value)
    except ValueError:
        return False
    if parts.scheme not in ("http", "https"):
        return False
    return (parts.hostname or "").lower() in _LOCAL_HOSTS


class LocalOnlyMiddleware(BaseHTTPMiddleware):
    """Reject non-localhost Host and cross-site Origin/Referer headers."""

    async def dispatch(self, request, call_next):
        if not _host_ok(request.headers.get("host", "")):
            return JSONResponse({"detail": "Forbidden: non-local Host"},
                                status_code=403)
        if not _origin_ok(request.headers.get("origin")):
            return JSONResponse({"detail": "Forbidden: cross-site Origin"},
                                status_code=403)
        if not _origin_ok(request.headers.get("referer")):
            return JSONResponse({"detail": "Forbidden: cross-site Referer"},
                                status_code=403)
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = _CSP
        return response


@lru_cache(maxsize=8)
def _have_module(name: str) -> bool:
    """Import-system probe that executes no module code (cheap, cached)."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    logbuffer.install()
    if get_settings().mcp_autostart:
        try:
            get_mcp_manager().start()
        except Exception:  # autostart is best-effort; the tab shows the error
            pass
    yield
    get_mcp_manager().stop()
    get_job_manager().shutdown()


app = FastAPI(title="procure", lifespan=lifespan)
app.add_middleware(LocalOnlyMiddleware)

_DEFAULT_TOPK = get_settings().top_k

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
INDEX_HTML = os.path.join(STATIC_DIR, "index.html")
ASSETS_DIR = os.path.join(STATIC_DIR, "assets")


def _safe_detail(message: str) -> str:
    """Redact the absolute data dir from user-facing error details."""
    data_dir = get_settings().data_dir
    return str(message).replace(data_dir, "<data-dir>") if data_dir else str(message)


class SearchResponse(BaseModel):
    query: str
    results: list[dict]


class SettingsPatch(BaseModel):
    search: str | None = None
    rerank: str | None = None
    mcp_autostart: bool | None = None


class TagsPatch(BaseModel):
    tags: list[str] = []


class MetaPatch(BaseModel):
    filename: str | None = None
    doc_type: str | None = None


class SkillsInstall(BaseModel):
    force: bool = False
    targets: list[str] | None = None


@app.get("/api/health")
def health() -> dict:
    svc = get_service()
    neural = _have_module("sentence_transformers") and _have_module("torch")
    return {
        "status": "ok",
        "version": __version__,
        "embed_dim": svc.embedder.dim,
        "chunks": svc.vectors.count(),
        "documents": svc.meta.count(),
        "search": svc.settings.search_mode,
        "rerank": svc.settings.rerank,
        # Cheap flags only: .available would LOAD the NLI model on first
        # call, and the frontend polls health after every action.
        "rerank_backend": "neural" if neural else "heuristic",
        "rerank_models_loaded": {
            "cross_encoder": svc._cross.loaded,
            "nli": svc._nli.loaded,
        },
        "mcp_autostart": svc.settings.mcp_autostart,
        "dense": svc.dimension_status(),
        "embedding": svc.embedding_status(),
        "ocr": ocr_backend(),
        "formats": sorted(extract.SUPPORTED_EXTENSIONS),
    }


@app.post("/api/documents/upload")
def upload(files: list[UploadFile] = File(...),
           background: bool = Query(False, alias="async"),
           doc_type: str = Query("document")) -> dict:
    """Ingest uploaded files. ?async=1 runs as a background job instead."""
    from app.store import normalize_doc_type

    try:
        normalize_doc_type(doc_type)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    svc = get_service()
    # Chunked read with a per-file cap: never buffer an unbounded upload.
    limit = max(1, svc.settings.max_upload_mb) * 1024 * 1024
    payload: list[tuple[str, bytes]] = []
    for f in files:
        chunks: list[bytes] = []
        size = 0
        while True:
            piece = f.file.read(1024 * 1024)
            if not piece:
                break
            size += len(piece)
            if size > limit:
                raise HTTPException(
                    413, f"{f.filename or 'unnamed'} exceeds the "
                    f"{svc.settings.max_upload_mb} MB per-file limit")
            chunks.append(piece)
        payload.append((f.filename or "unnamed", b"".join(chunks)))
    if not background:
        return {"documents": svc.ingest_files(payload, doc_type=doc_type)}
    jobs = get_job_manager()
    job = jobs.submit(
        "upload", f"{len(payload)} file(s)",
        lambda h: svc.ingest_files(
            payload, doc_type=doc_type,
            progress=h.update, cancelled=h.is_cancelled),
    )
    return {"job": job}


@app.get("/api/documents")
def list_documents(doc_type: str | None = Query(None),
                   q: str | None = Query(None),
                   limit: int | None = Query(None, ge=1, le=500),
                   offset: int = Query(0, ge=0),
                   sort: str = Query("newest")) -> dict:
    try:
        svc = get_service()
        return {"documents": svc.list_documents(doc_type, q,
                                                limit, offset, sort),
                "total": svc.count_documents(doc_type, q)}
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@app.get("/api/library/version")
def library_version() -> dict:
    """Cheap revision fingerprint for UI polling.

    Two aggregate SQLite queries, no vector-store access: the library
    tab polls this every few seconds and reloads only when it moves,
    so writes from the MCP server (or any other process sharing the
    data dir) show up without a manual refresh.
    """
    return get_service().library_version()


@app.get("/api/documents/export")
def export_library() -> Response:
    import json

    body = json.dumps(get_service().export_library(), ensure_ascii=False)
    return Response(
        content=body, media_type="application/json",
        headers={"Content-Disposition":
                 "attachment; filename=procure-export.json"})


@app.post("/api/documents/import")
async def import_library(file: UploadFile = File(...)) -> dict:
    import json

    try:
        payload = json.loads((await file.read()).decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as e:
        raise HTTPException(400, f"Not a procure export file: {e}") from e
    svc = get_service()
    job = get_job_manager().submit(
        "import", file.filename or "import",
        lambda h: svc.import_library(payload, progress=h.update,
                                     cancelled=h.is_cancelled),
    )
    return {"job": job}


@app.get("/api/documents/{doc_id}")
def get_document(doc_id: str) -> dict:
    try:
        return get_service().get_document(doc_id)
    except KeyError:
        raise HTTPException(404, f"Unknown document {doc_id}") from None


@app.patch("/api/documents/{doc_id}")
def patch_document(doc_id: str, body: MetaPatch) -> dict:
    if body.filename is None and body.doc_type is None:
        raise HTTPException(400, "Nothing to update: pass filename and/or doc_type")
    try:
        return get_service().update_document_meta(
            doc_id, body.filename, body.doc_type)
    except KeyError:
        raise HTTPException(404, f"Unknown document {doc_id}") from None
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str) -> dict:
    if not get_service().delete_document(doc_id):
        raise HTTPException(404, f"Unknown document {doc_id}")
    return {"deleted": doc_id}


@app.post("/api/documents/{doc_id}/reingest")
def reingest_document(doc_id: str,
                      background: bool = Query(False, alias="async")) -> dict:
    svc = get_service()
    if not background:
        try:
            return svc.reingest_document(doc_id)
        except KeyError:
            raise HTTPException(404, f"Unknown document {doc_id}") from None
        except FileNotFoundError as e:
            raise HTTPException(410, _safe_detail(str(e))) from e
    try:
        svc.meta.get(doc_id) or (_ for _ in ()).throw(KeyError(doc_id))
    except KeyError:
        raise HTTPException(404, f"Unknown document {doc_id}") from None
    job = get_job_manager().submit(
        "reingest", doc_id,
        lambda h: [svc.reingest_document(
            doc_id,
            progress_frac=lambda frac, stage: h.update(frac, 1, stage))])
    return {"job": job}


@app.patch("/api/documents/{doc_id}/tags")
def patch_tags(doc_id: str, body: TagsPatch) -> dict:
    try:
        return get_service().set_tags(doc_id, body.tags)
    except KeyError:
        raise HTTPException(404, f"Unknown document {doc_id}") from None


@app.get("/api/tags")
def list_tags() -> dict:
    return {"tags": get_service().list_tags()}


@app.get("/api/jobs")
def list_jobs() -> dict:
    return {"jobs": get_job_manager().list()}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = get_job_manager().get(job_id)
    if job is None:
        raise HTTPException(404, f"Unknown job {job_id}")
    return job


@app.delete("/api/jobs/{job_id}")
def cancel_job(job_id: str) -> dict:
    job = get_job_manager().cancel(job_id)
    if job is None:
        raise HTTPException(404, f"Unknown job {job_id}")
    return job


@app.get("/api/search", response_model=SearchResponse)
def search(q: str = Query(""), top_k: int = Query(_DEFAULT_TOPK, ge=1, le=50),
           doc_id: list[str] | None = Query(None),
           tag: list[str] | None = Query(None),
           doc_type: str | None = Query(None),
           since: str | None = Query(None)) -> dict:
    try:
        return {"query": q, "results": get_service().search(
            q, top_k, doc_id, tag, doc_type, since)}
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@app.patch("/api/settings")
def patch_settings(patch: SettingsPatch) -> dict:
    svc = get_service()
    out: dict = {}
    try:
        if patch.search is not None or patch.rerank is not None:
            out.update(svc.update_settings(patch.search, patch.rerank))
        if patch.mcp_autostart is not None:
            out.update(svc.set_mcp_autostart(patch.mcp_autostart))
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    if not out:
        raise HTTPException(400, "Nothing to update")
    out["mcp_autostart"] = svc.settings.mcp_autostart
    return out


@app.get("/api/mcp/status")
def mcp_status() -> dict:
    return get_mcp_manager().status()


@app.post("/api/mcp/start")
def mcp_start() -> dict:
    return get_mcp_manager().start()


@app.post("/api/mcp/stop")
def mcp_stop() -> dict:
    return get_mcp_manager().stop()


@app.get("/api/mcp/logs")
def mcp_logs(since: int = Query(0, ge=0)) -> dict:
    return get_mcp_manager().logs(since)


@app.get("/api/mcp/tools")
def mcp_tools() -> dict:
    """Tool name + description, single-sourced from the MCP server."""
    from app import mcp_server

    return {"tools": mcp_server.tool_descriptions()}


@app.get("/api/logs")
def workspace_logs(since: int = Query(0, ge=0)) -> dict:
    return logbuffer.get_buffer().read(since)


@app.get("/api/skills/status")
def skills_status() -> dict:
    from app import skills

    return skills.status()


@app.post("/api/skills/install")
def skills_install(body: SkillsInstall) -> dict:
    from app import skills

    try:
        return skills.install(body.targets, body.force)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(500, str(e)) from e


if os.path.isdir(ASSETS_DIR):
    app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")

    if os.path.isfile(INDEX_HTML):
        @app.get("/", include_in_schema=False)
        def index() -> FileResponse:
            return FileResponse(INDEX_HTML)

        @app.get("/{path:path}", include_in_schema=False)
        def spa_fallback(path: str):
            # API routes match before this catch-all; unknown non-API paths
            # serve the SPA instead of a bare 404.
            if path.startswith("api/"):
                raise HTTPException(404, "Not found")
            return FileResponse(INDEX_HTML)
