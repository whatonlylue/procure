"""FastAPI app: /api/* + static frontend."""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.config import get_settings
from app.mcp_manager import get_mcp_manager
from app.service import get_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    get_mcp_manager().stop()


app = FastAPI(title="procure", lifespan=lifespan)

_DEFAULT_TOPK = get_settings().top_k

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


class SearchResponse(BaseModel):
    query: str
    results: list[dict]


class SettingsPatch(BaseModel):
    search: str | None = None
    rerank: str | None = None


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
    }


@app.post("/api/documents/upload")
async def upload(files: list[UploadFile] = File(...)) -> dict:
    svc = get_service()
    payload = [(f.filename or "unnamed", await f.read()) for f in files]
    results = svc.ingest_files(payload)
    return {"documents": results}


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


@app.get("/api/search", response_model=SearchResponse)
def search(q: str = Query(""), top_k: int = Query(_DEFAULT_TOPK, ge=1, le=50),
           doc_id: list[str] | None = Query(None)) -> dict:
    return {"query": q, "results": get_service().search(q, top_k, doc_id)}


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
