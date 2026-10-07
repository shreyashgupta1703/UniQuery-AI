"""API tests for the streaming endpoint and the serving-hardening layer.

Exercises the SSE ask/stream protocol end-to-end and the security features
(upload size cap, extension allowlist, optional bearer auth, rate limiting) via
FastAPI's TestClient — no server, no network.
"""

from __future__ import annotations

import importlib
import json

import pytest

pytest.importorskip("httpx")
pytest.importorskip("starlette.testclient")


def _make_client(monkeypatch, **env):
    monkeypatch.setenv("LOCALRAG_DB", ":memory:")
    monkeypatch.setenv("LOCALRAG_DISABLE_OLLAMA", "1")
    for k, v in env.items():
        monkeypatch.setenv(k, v)

    import app.config as config

    importlib.reload(config)
    import app.engine as engine

    importlib.reload(engine)
    import app.security as security

    importlib.reload(security)
    import app.main as main

    importlib.reload(main)

    from starlette.testclient import TestClient

    return TestClient(main.app)


@pytest.fixture()
def client(monkeypatch):
    with _make_client(monkeypatch) as c:
        yield c


def _ingest(client):
    doc = (
        "Solar panels convert sunlight into electricity using photovoltaic "
        "cells. Commercial silicon panels reach 18 to 22 percent efficiency."
    )
    r = client.post("/api/ingest/text", json={"text": doc, "source": "solar.md"})
    assert r.status_code == 200, r.text


# ------------------------------------------------------------------- streaming
def test_ask_stream_emits_meta_tokens_citations(client):
    _ingest(client)
    events = []
    with client.stream(
        "POST", "/api/ask/stream", json={"question": "How efficient are panels?"}
    ) as resp:
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        for line in resp.iter_lines():
            if not line:
                continue
            line = line[6:] if line.startswith("data: ") else line
            events.append(json.loads(line))

    types = [e["type"] for e in events]
    assert types[0] == "meta"
    assert "token" in types
    assert "citations" in types
    assert types[-1] == "done"

    text = "".join(e["text"] for e in events if e["type"] == "token")
    assert text.strip()
    cite_event = next(e for e in events if e["type"] == "citations")
    assert cite_event["citations"][0]["source"] == "solar.md"


def test_ask_returns_grounded_flag(client):
    _ingest(client)
    r = client.post("/api/ask", json={"question": "How efficient are panels?"})
    assert r.status_code == 200
    assert r.json()["grounded"] is True


# --------------------------------------------------------------- ingest guards
def test_rejects_unsupported_extension(client):
    files = {"file": ("evil.exe", b"MZ binary", "application/octet-stream")}
    r = client.post("/api/ingest/file", files=files)
    assert r.status_code == 400
    assert "unsupported" in r.json()["detail"].lower()


def test_accepts_allowed_extension(client):
    files = {"file": ("notes.md", b"# Notes\n\nSome content about testing.", "text/markdown")}
    r = client.post("/api/ingest/file", files=files)
    assert r.status_code == 200
    assert r.json()["source"] == "notes.md"


def test_upload_size_cap_enforced(monkeypatch):
    with _make_client(monkeypatch, LOCALRAG_MAX_UPLOAD_BYTES="64") as client:
        big = b"x" * 500
        files = {"file": ("big.txt", big, "text/plain")}
        r = client.post("/api/ingest/file", files=files)
        assert r.status_code == 413


def test_empty_file_rejected(client):
    files = {"file": ("empty.txt", b"", "text/plain")}
    r = client.post("/api/ingest/file", files=files)
    assert r.status_code == 400


# ---------------------------------------------------------------------- auth
def test_auth_required_when_token_set(monkeypatch):
    with _make_client(monkeypatch, LOCALRAG_API_TOKEN="secret123") as client:
        # Health stays public.
        assert client.get("/api/health").status_code == 200
        # Protected route without a token -> 401.
        r = client.get("/api/stats")
        assert r.status_code == 401
        # Wrong token -> 401.
        r = client.get("/api/stats", headers={"Authorization": "Bearer nope"})
        assert r.status_code == 401
        # Correct token -> 200.
        r = client.get("/api/stats", headers={"Authorization": "Bearer secret123"})
        assert r.status_code == 200
        # CORS preflight (OPTIONS) must not be blocked by auth.
        r = client.options(
            "/api/ask",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert r.status_code < 400


# ------------------------------------------------------------------ rate limit
def test_rate_limit_returns_429(monkeypatch):
    with _make_client(
        monkeypatch, LOCALRAG_RATE_LIMIT="3", LOCALRAG_RATE_WINDOW="60"
    ) as client:
        codes = [client.get("/api/health").status_code for _ in range(5)]
        assert codes.count(200) == 3
        assert codes.count(429) == 2
