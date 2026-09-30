"""FastAPI app: /api/* + static frontend."""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import extract
from app.config import get_settings
from app.jobs import get_job_manager
from app.mcp_manager import get_mcp_manager
from app.ocr import backend_name as ocr_backend
from app.service import get_service
from app.watch import get_watch_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_watch_manager().ensure_running()
    yield
    get_watch_manager().stop()
    get_mcp_manager().stop()
    get_job_manager().shutdown()


app = FastAPI(title="procure", lifespan=lifespan)

_DEFAULT_TOPK = get_settings().top_k

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


class SearchResponse(BaseModel):
    query: str
    results: list[dict]


class SettingsPatch(BaseModel):
    search: str | None = None
    rerank: str | None = None


class UrlIngest(BaseModel):
    url: str
    tags: list[str] = []


class TagsPatch(BaseModel):
    tags: list[str] = []


class WatchAdd(BaseModel):
    path: str
    recursive: bool = True


@app.get("/api/health")
def health() -> dict:
    svc = get_service()
    return {
        "status": "ok",
        "embeddings": svc.settings.embeddings,
        "vectordb": svc.settings.vectordb,
        "embed_dim": svc.embedder.dim,
        "chunks": svc.vectors.count(),
        "documents": len(svc.meta.list()),
        "search": svc.settings.search_mode,
        "rerank": svc.settings.rerank,
        "nli_available": svc._nli.available,
        "alpha_nli": svc.settings.alpha_nli,
        "ocr": ocr_backend(),
        "formats": sorted(extract.SUPPORTED_EXTENSIONS),
    }


@app.post("/api/documents/upload")
async def upload(files: list[UploadFile] = File(...),
                 background: bool = Query(False, alias="async")) -> dict:
    """Ingest uploaded files. ?async=1 runs as a background job instead."""
    svc = get_service()
    payload = [(f.filename or "unnamed", await f.read()) for f in files]
    if not background:
        return {"documents": svc.ingest_files(payload)}
    jobs = get_job_manager()
    job = jobs.submit(
        "upload", f"{len(payload)} file(s)",
        lambda h: svc.ingest_files(
            payload, progress=h.update, cancelled=h.is_cancelled),
    )
    return {"job": job}


@app.post("/api/documents/url")
def ingest_url(body: UrlIngest) -> dict:
    return get_service().ingest_url(body.url, body.tags)


@app.get("/api/documents")
def list_documents() -> dict:
    return {"documents": get_service().list_documents()}


@app.get("/api/documents/{doc_id}")
def get_document(doc_id: str) -> dict:
    try:
        return get_service().get_document(doc_id)
    except KeyError:
        raise HTTPException(404, f"Unknown document {doc_id}") from None


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str) -> dict:
    if not get_service().delete_document(doc_id):
        raise HTTPException(404, f"Unknown document {doc_id}")
    return {"deleted": doc_id}


@app.post("/api/documents/{doc_id}/reingest")
def reingest_document(doc_id: str) -> dict:
    try:
        return get_service().reingest_document(doc_id)
    except KeyError:
        raise HTTPException(404, f"Unknown document {doc_id}") from None
    except FileNotFoundError as e:
        raise HTTPException(410, str(e)) from e


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


@app.get("/api/watch")
def list_watches() -> dict:
    mgr = get_watch_manager()
    return {"folders": mgr.list(), "last_scan": mgr.last_scan}


@app.post("/api/watch")
def add_watch(body: WatchAdd) -> dict:
    try:
        return get_watch_manager().add(body.path, body.recursive)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@app.delete("/api/watch/{watch_id}")
def remove_watch(watch_id: int) -> dict:
    if not get_watch_manager().remove(watch_id):
        raise HTTPException(404, f"Unknown watch {watch_id}")
    return {"removed": watch_id}


@app.post("/api/watch/sync")
def sync_watches() -> dict:
    mgr = get_watch_manager()
    job = get_job_manager().submit(
        "watch-sync", "folder sync",
        lambda h: mgr.sync_all(progress=h.update, cancelled=h.is_cancelled),
    )
    return {"job": job}


@app.get("/api/search", response_model=SearchResponse)
def search(q: str = Query(""), top_k: int = Query(_DEFAULT_TOPK, ge=1, le=50),
           doc_id: list[str] | None = Query(None),
           tag: list[str] | None = Query(None),
           source: str | None = Query(None)) -> dict:
    return {"query": q, "results": get_service().search(q, top_k, doc_id, tag, source)}


@app.patch("/api/settings")
def patch_settings(patch: SettingsPatch) -> dict:
    try:
        return get_service().update_settings(patch.search, patch.rerank)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


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


if os.path.isdir(STATIC_DIR):
    app.mount("/assets", StaticFiles(directory=os.path.join(STATIC_DIR, "assets")), name="assets")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(os.path.join(STATIC_DIR, "index.html"))
