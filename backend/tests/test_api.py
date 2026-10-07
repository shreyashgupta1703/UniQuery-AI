"""API smoke tests using FastAPI's TestClient (no server, no network)."""

from __future__ import annotations

import importlib

import pytest

# Skip gracefully if optional test deps are missing; core tests still run.
httpx = pytest.importorskip("httpx")
starlette_testclient = pytest.importorskip("starlette.testclient")


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALRAG_DB", ":memory:")
    monkeypatch.setenv("LOCALRAG_DISABLE_OLLAMA", "1")
    # Reload config + engine + main so the env vars take effect on a clean app.
    import app.config as config

    importlib.reload(config)
    import app.engine as engine

    importlib.reload(engine)
    import app.main as main

    importlib.reload(main)

    from starlette.testclient import TestClient

    with TestClient(main.app) as c:
        yield c


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_ingest_and_ask_flow(client):
    doc = (
        "Mars is the fourth planet from the Sun. It is known as the Red Planet "
        "because iron oxide on its surface gives it a reddish appearance. "
        "Mars has two small moons named Phobos and Deimos."
    )
    r = client.post("/api/ingest/text", json={"text": doc, "source": "mars.md"})
    assert r.status_code == 200, r.text
    assert r.json()["n_chunks"] >= 1

    r = client.post("/api/ask", json={"question": "Why is Mars called the Red Planet?"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["mode"] == "extractive"
    assert body["citations"]
    assert body["citations"][0]["source"] == "mars.md"


def test_stats_endpoint(client):
    client.post("/api/ingest/text", json={"text": "hello world doc", "source": "h.md"})
    r = client.get("/api/stats")
    assert r.status_code == 200
    body = r.json()
    assert body["embedder"]
    assert body["ollama_available"] is False


def test_delete_missing_source_404(client):
    r = client.delete("/api/sources/does-not-exist.md")
    assert r.status_code == 404


def test_ingest_empty_text_rejected(client):
    r = client.post("/api/ingest/text", json={"text": "", "source": "x.md"})
    assert r.status_code == 422  # pydantic min_length validation
