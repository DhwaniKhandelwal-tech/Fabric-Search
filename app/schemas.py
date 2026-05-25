"""
Pydantic models for API request/response validation.
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class FabricResult(BaseModel):
    """A single search result."""
    id: str = Field(..., description="Unique image identifier (filename stem)")
    filename: str = Field(..., description="Original filename")
    url: str = Field(..., description="Public URL of the image")
    tags: str = Field(..., description="Comma-separated tags inferred from filename")
    score: float = Field(..., description="Cosine similarity score (0–1, higher = more similar)")


class SearchResponse(BaseModel):
    """Response for both image-search and text-search endpoints."""
    query_type: str = Field(..., description="'image' or 'text'")
    results: List[FabricResult]
    total: int = Field(..., description="Number of results returned")


class HealthResponse(BaseModel):
    status: str
    indexed_images: int
    model: str


class StatsResponse(BaseModel):
    total_indexed: int
    lancedb_uri: str
    clip_model: str
    embedding_dim: int
