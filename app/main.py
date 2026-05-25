"""
Fabric Search API
=================
FastAPI application exposing:

    POST /api/v1/search/image   — find similar fabrics by uploading an image
    POST /api/v1/search/text    — find fabrics by text description
    GET  /api/v1/images/{id}    — retrieve image metadata by ID
    GET  /api/v1/health         — health check
    GET  /api/v1/stats          — index statistics

Run locally:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import logging
from typing import Annotated, Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import (
    API_DESCRIPTION,
    API_TITLE,
    API_VERSION,
    CLIP_MODEL_NAME,
    DEFAULT_TOP_K,
    EMBEDDING_DIM,
    LANCEDB_URI,
    MAX_TOP_K,
)
from app.embeddings import get_embedding_model
from app.schemas import FabricResult, HealthResponse, SearchResponse, StatsResponse
from app.store import count_rows, get_or_create_table, vector_search

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ── App setup ──────────────────────────────────────────────────────────────

app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pre-load model at startup (avoid cold-start on first request)
@app.on_event("startup")
async def _preload_model() -> None:
    logger.info("Pre-loading CLIP embedding model …")
    get_embedding_model()
    logger.info("Model ready. Indexed images: %d", count_rows())


# ── Helpers ────────────────────────────────────────────────────────────────

_ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/gif",
}

def _validate_image_upload(file: UploadFile) -> None:
    if file.content_type and file.content_type not in _ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type '{file.content_type}'. "
                   f"Accepted: {sorted(_ALLOWED_CONTENT_TYPES)}",
        )


def _rows_to_results(rows: list) -> list[FabricResult]:
    return [
        FabricResult(
            id=r["id"],
            filename=r["filename"],
            url=r["url"],
            tags=r.get("tags", ""),
            score=r.get("score", 0.0),
        )
        for r in rows
    ]


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/api/v1/health", response_model=HealthResponse, tags=["meta"])
async def health() -> HealthResponse:
    """Service health check — also returns total indexed images."""
    return HealthResponse(
        status="ok",
        indexed_images=count_rows(),
        model=CLIP_MODEL_NAME,
    )


@app.get("/api/v1/stats", response_model=StatsResponse, tags=["meta"])
async def stats() -> StatsResponse:
    """Return index statistics."""
    return StatsResponse(
        total_indexed=count_rows(),
        lancedb_uri=LANCEDB_URI,
        clip_model=CLIP_MODEL_NAME,
        embedding_dim=EMBEDDING_DIM,
    )


@app.post("/api/v1/search/image", response_model=SearchResponse, tags=["search"])
async def search_by_image(
    file: Annotated[UploadFile, File(description="Fabric image to search with")],
    top_k: Annotated[
        int,
        Query(ge=1, le=MAX_TOP_K, description="Number of results to return"),
    ] = DEFAULT_TOP_K,
) -> SearchResponse:
    """
    Upload a fabric image and get the **top_k** most visually similar fabrics.

    - Accepts JPEG, PNG, WebP
    - Returns results ordered by cosine similarity (highest first)
    """
    _validate_image_upload(file)

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    try:
        model = get_embedding_model()
        query_vec = model.embed_image_bytes(image_bytes)
    except Exception as exc:
        logger.exception("Embedding failed for uploaded image")
        raise HTTPException(status_code=422, detail=f"Could not process image: {exc}")

    rows = vector_search(query_vec, top_k=top_k)
    results = _rows_to_results(rows)

    return SearchResponse(query_type="image", results=results, total=len(results))


@app.post("/api/v1/search/text", response_model=SearchResponse, tags=["search"])
async def search_by_text(
    query: Annotated[
        str,
        Form(description="Natural language description, e.g. 'blue floral cotton'"),
    ],
    top_k: Annotated[
        int,
        Query(ge=1, le=MAX_TOP_K, description="Number of results to return"),
    ] = DEFAULT_TOP_K,
) -> SearchResponse:
    """
    Search fabrics using a **text description**.

    CLIP understands concepts like:
    - `"red and white plaid"`
    - `"silk with geometric pattern"`
    - `"dark denim texture"`
    """
    query = query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")

    try:
        model = get_embedding_model()
        query_vec = model.embed_text(query)
    except Exception as exc:
        logger.exception("Text embedding failed")
        raise HTTPException(status_code=422, detail=f"Could not embed query: {exc}")

    rows = vector_search(query_vec, top_k=top_k)
    results = _rows_to_results(rows)

    return SearchResponse(query_type="text", results=results, total=len(results))


@app.get("/api/v1/images/{image_id}", response_model=FabricResult, tags=["images"])
async def get_image(image_id: str) -> FabricResult:
    """Retrieve metadata for a single indexed image by its ID."""
    table = get_or_create_table()
    try:
        rows = (
            table.search()
            .where(f"id = '{image_id}'")
            .limit(1)
            .to_list()
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    if not rows:
        raise HTTPException(status_code=404, detail=f"Image '{image_id}' not found.")

    r = rows[0]
    return FabricResult(
        id=r["id"],
        filename=r["filename"],
        url=r["url"],
        tags=r.get("tags", ""),
        score=1.0,
    )


# ── Root redirect ──────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    return JSONResponse({"message": "Fabric Search API", "docs": "/docs"})
