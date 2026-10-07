"""Pydantic request/response models for the API."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class IngestTextRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Raw document text")
    source: str = Field(..., min_length=1, description="A name/label for the source")


class IngestResponse(BaseModel):
    source: str
    n_chunks: int
    document_id: int


class CitationModel(BaseModel):
    marker: int
    source: str
    chunk_index: int
    score: float
    snippet: str


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: Optional[int] = Field(default=None, ge=1, le=20)
    source: Optional[str] = Field(default=None, description="Restrict to one source")


class AskResponse(BaseModel):
    answer: str
    mode: str
    model: str = ""
    grounded: bool = True
    citations: List[CitationModel] = []


class SourceInfo(BaseModel):
    source: str
    n_chunks: int
    created_at: Optional[str] = None


class StatsResponse(BaseModel):
    embedder: str
    embed_dim: Optional[int] = None
    n_chunks: int
    ollama_available: bool
    sources: List[SourceInfo] = []
