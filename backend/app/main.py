"""FastAPI application exposing the localrag engine.

Endpoints:
    GET  /api/health          - liveness + backend mode
    GET  /api/stats           - embedder, chunk count, sources, ollama status
    POST /api/ingest/text     - ingest raw text
    POST /api/ingest/file     - ingest an uploaded file (md/txt/pdf)
    POST /api/ask             - ask a question, get a cited answer (blocking)
    POST /api/ask/stream      - ask a question, stream the answer as SSE
    DELETE /api/sources/{name}- remove a source and its chunks

Serving is hardened for LAN exposure: a configurable CORS allowlist, an optional
bearer-token gate, a fixed-window rate limit, and a size-capped, streamed file
upload path. All of it is off/permissive-by-default so the zero-config local
experience is unchanged.

If a production frontend build exists at ``frontend/dist`` it is served at ``/``.
The module imports cleanly with zero optional deps installed, which is what the
CI smoke test relies on.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from .config import settings
from .engine import RagEngine
from .generation import ollama_available
from .schemas import (
    AskRequest,
    AskResponse,
    CitationModel,
    IngestResponse,
    IngestTextRequest,
    SourceInfo,
    StatsResponse,
)
from .security import AuthMiddleware, RateLimitMiddleware

app = FastAPI(
    title="UniQuery AI",
    version="0.2.0",
    description="Offline university knowledge assistant with source-grounded answers and local LLMs.",
)

# CORS is allowlist-driven. "*" is honoured but discouraged; the default is the
# Vite dev origin so a fresh checkout "just works" without opening the app up.
_cors_origins = settings.cors_origins
_allow_all = _cors_origins == ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=not _allow_all,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)
# Optional auth + rate limit. Both no-op unless configured (auth) / positive
# (rate limit), preserving the zero-config path.
app.add_middleware(AuthMiddleware, token=settings.api_token)
app.add_middleware(
    RateLimitMiddleware,
    limit=settings.rate_limit,
    window_seconds=settings.rate_window_seconds,
)

# One engine for the app's lifetime. Created lazily so importing the module
# (e.g. in CI) never triggers a model download or DB write until first use.
_engine: RagEngine | None = None


def get_engine() -> RagEngine:
    global _engine
    if _engine is None:
        _engine = RagEngine()
    return _engine


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "version": app.version}


@app.get("/api/stats", response_model=StatsResponse)
def stats() -> StatsResponse:
    eng = get_engine()
    s = eng.stats()
    return StatsResponse(
        embedder=s["embedder"],
        embed_dim=s["embed_dim"],
        n_chunks=s["n_chunks"],
        ollama_available=ollama_available(),
        sources=[SourceInfo(**src) for src in s["sources"]],
    )


@app.post("/api/ingest/text", response_model=IngestResponse)
def ingest_text(req: IngestTextRequest) -> IngestResponse:
    eng = get_engine()
    try:
        result = eng.ingest_text(req.text, req.source)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return IngestResponse(
        source=result.source,
        n_chunks=result.n_chunks,
        document_id=result.document_id,
    )


@app.post("/api/ingest/file", response_model=IngestResponse)
async def ingest_file(file: UploadFile = File(...)) -> IngestResponse:
    eng = get_engine()
    filename = file.filename or "upload.txt"
    suffix = Path(filename).suffix.lower() or ".txt"

    allowed = {e.lower() for e in settings.allowed_extensions}
    if suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"unsupported file type '{suffix}'. Allowed: {sorted(allowed)}",
        )

    # Stream the upload to a temp file in bounded chunks, enforcing the size cap
    # without ever holding the whole payload in memory.
    limit = settings.max_upload_bytes
    total = 0
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = Path(tmp.name)
            while True:
                block = await file.read(1024 * 256)
                if not block:
                    break
                total += len(block)
                if total > limit:
                    raise HTTPException(
                        status_code=413,
                        detail=f"file exceeds max upload size of {limit} bytes",
                    )
                tmp.write(block)
        if total == 0:
            raise HTTPException(status_code=400, detail="empty file")

        try:
            result = eng.ingest_file(tmp_path)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)

    # Preserve the original filename as the source label.
    return IngestResponse(
        source=filename or result.source,
        n_chunks=result.n_chunks,
        document_id=result.document_id,
    )


@app.post("/api/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    eng = get_engine()
    ans = eng.answer(req.question, top_k=req.top_k, source=req.source)
    return AskResponse(
        answer=ans.answer,
        mode=ans.mode,
        model=ans.model,
        grounded=ans.grounded,
        citations=[
            CitationModel(
                marker=c.marker,
                source=c.source,
                chunk_index=c.chunk_index,
                score=c.score,
                snippet=c.snippet,
            )
            for c in ans.citations
        ],
    )


@app.post("/api/ask/stream")
def ask_stream(req: AskRequest) -> StreamingResponse:
    """Stream the answer as Server-Sent Events.

    Emits one ``data:`` line per JSON event:
      * ``{"type":"meta", ...}``      - mode/model/grounded (first).
      * ``{"type":"token","text":…}`` - incremental answer text.
      * ``{"type":"citations", ...}`` - resolved citations (last).
      * ``{"type":"done"}``           - terminal sentinel.

    Works for both the Ollama (live tokens) and extractive (word-streamed)
    paths, so the client renders the answer live regardless of backend mode.
    """

    eng = get_engine()

    async def event_source() -> AsyncIterator[bytes]:
        # The engine's stream generator is synchronous; iterate it directly.
        # Each event is small, so this stays responsive without a threadpool.
        for event in eng.answer_stream(
            req.question, top_k=req.top_k, source=req.source
        ):
            payload = {"type": event.type, **event.data}
            yield f"data: {json.dumps(payload)}\n\n".encode("utf-8")
        yield b'data: {"type": "done"}\n\n'

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.delete("/api/sources/{name}")
def delete_source(name: str) -> JSONResponse:
    eng = get_engine()
    removed = eng.store.delete_source(name)
    if removed == 0:
        raise HTTPException(status_code=404, detail=f"no such source: {name}")
    return JSONResponse({"deleted": name, "documents_removed": removed})


# --------------------------------------------------------------------- static
# Serve the built React app if present (frontend/dist). Mounted last so it does
# not shadow the /api routes.
_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if _DIST.is_dir():
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=str(_DIST), html=True), name="frontend")


def main() -> None:  # pragma: no cover - manual entrypoint
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":  # pragma: no cover
    main()
