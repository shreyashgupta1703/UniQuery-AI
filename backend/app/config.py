"""Runtime configuration for localrag.

Everything is environment-driven so the whole stack runs with zero config out of
the box, but can be pointed at a real Ollama server / different embedding model
without code changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_list(name: str, default: list[str]) -> list[str]:
    raw = os.getenv(name)
    if raw is None:
        return list(default)
    items = [x.strip() for x in raw.split(",") if x.strip()]
    return items or list(default)


@dataclass
class Settings:
    """Central settings object, populated from the environment."""

    # Where the SQLite vector store lives. ":memory:" is honoured for tests.
    db_path: str = field(
        default_factory=lambda: os.getenv(
            "LOCALRAG_DB", str(Path(__file__).resolve().parents[2] / "localrag.db")
        )
    )

    # Embedding model name (sentence-transformers). If the library or model is
    # unavailable we transparently fall back to a hashing vectorizer.
    embed_model: str = field(
        default_factory=lambda: os.getenv(
            "LOCALRAG_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )
    # Dimensionality of the deterministic hashing-vectorizer fallback.
    hash_dim: int = field(default_factory=lambda: _env_int("LOCALRAG_HASH_DIM", 512))

    # Chunking parameters (in characters).
    chunk_size: int = field(default_factory=lambda: _env_int("LOCALRAG_CHUNK_SIZE", 900))
    chunk_overlap: int = field(
        default_factory=lambda: _env_int("LOCALRAG_CHUNK_OVERLAP", 150)
    )

    # Retrieval defaults.
    top_k: int = field(default_factory=lambda: _env_int("LOCALRAG_TOP_K", 4))

    # Ollama generation endpoint. If unreachable we fall back to an extractive
    # answer built directly from the retrieved chunks.
    ollama_url: str = field(
        default_factory=lambda: os.getenv(
            "LOCALRAG_OLLAMA_URL", "http://localhost:11434"
        )
    )
    ollama_model: str = field(
        default_factory=lambda: os.getenv("LOCALRAG_OLLAMA_MODEL", "llama3.2")
    )
    ollama_timeout: int = field(
        default_factory=lambda: _env_int("LOCALRAG_OLLAMA_TIMEOUT", 60)
    )
    # Force the extractive fallback even if Ollama happens to be reachable.
    disable_ollama: bool = field(
        default_factory=lambda: _env_bool("LOCALRAG_DISABLE_OLLAMA", False)
    )

    # ---------------------------------------------------------------- serving
    # CORS allowlist. Defaults to the Vite dev origin + same-origin production;
    # set LOCALRAG_CORS_ORIGINS="*" to allow any origin (not recommended).
    cors_origins: list = field(
        default_factory=lambda: _env_list(
            "LOCALRAG_CORS_ORIGINS",
            ["http://localhost:5173", "http://127.0.0.1:5173"],
        )
    )
    # Reject uploads larger than this (bytes). Default 10 MiB.
    max_upload_bytes: int = field(
        default_factory=lambda: _env_int("LOCALRAG_MAX_UPLOAD_BYTES", 10 * 1024 * 1024)
    )
    # Accepted upload extensions (comma-separated, leading dot optional).
    allowed_extensions: list = field(
        default_factory=lambda: [
            e if e.startswith(".") else f".{e}"
            for e in _env_list(
                "LOCALRAG_ALLOWED_EXTENSIONS", [".md", ".markdown", ".txt", ".pdf"]
            )
        ]
    )
    # Optional bearer token. When set, all /api routes (except health) require
    # `Authorization: Bearer <token>`. Empty = no auth (default, local use).
    api_token: str = field(
        default_factory=lambda: os.getenv("LOCALRAG_API_TOKEN", "")
    )
    # Simple in-process rate limit: max requests per client IP per window.
    # Set rate_limit=0 to disable.
    rate_limit: int = field(
        default_factory=lambda: _env_int("LOCALRAG_RATE_LIMIT", 120)
    )
    rate_window_seconds: int = field(
        default_factory=lambda: _env_int("LOCALRAG_RATE_WINDOW", 60)
    )


settings = Settings()
