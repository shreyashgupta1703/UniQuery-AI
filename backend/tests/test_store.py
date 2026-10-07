"""Vector-store tests: search correctness, numpy/python equivalence, caching.

The store has two interchangeable search backends (a vectorized numpy path and a
pure-Python fallback). These tests assert they agree, that the in-memory matrix
cache is correctly invalidated on writes/deletes, and that source filtering and
top-k behave in both paths.
"""

from __future__ import annotations

import pytest

from app.chunking import chunk_text
from app.embeddings import HashingEmbedder
from app import store as store_mod
from app.store import VectorStore

DOCS = {
    "solar.md": (
        "Solar panels convert sunlight into electricity using photovoltaic "
        "cells. Commercial silicon panels reach 18 to 22 percent efficiency. "
        "Inverters convert direct current into alternating current."
    ),
    "coffee.md": (
        "Espresso is brewed by forcing hot water through finely ground coffee "
        "at nine bars of pressure. Pour-over gives a cleaner cup. Cold brew "
        "steeps coarse grounds for many hours."
    ),
}


@pytest.fixture()
def embedder():
    return HashingEmbedder(dim=256)


def _populate(store: VectorStore, embedder: HashingEmbedder) -> None:
    for name, text in DOCS.items():
        chunks = chunk_text(text, name, chunk_size=200, chunk_overlap=40)
        store.add_chunks(chunks, embedder.embed([c.text for c in chunks]))


def test_search_returns_top_k_sorted(embedder):
    store = VectorStore(":memory:")
    _populate(store, embedder)
    q = embedder.embed_one("photovoltaic efficiency of panels")
    hits = store.search(q, top_k=3)
    assert 1 <= len(hits) <= 3
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)
    assert hits[0].source == "solar.md"


def test_source_filter(embedder):
    store = VectorStore(":memory:")
    _populate(store, embedder)
    q = embedder.embed_one("energy")
    hits = store.search(q, top_k=5, source="coffee.md")
    assert hits
    assert all(h.source == "coffee.md" for h in hits)


def test_top_k_zero_returns_empty(embedder):
    store = VectorStore(":memory:")
    _populate(store, embedder)
    assert store.search(embedder.embed_one("anything"), top_k=0) == []


def test_numpy_and_python_paths_agree(embedder, monkeypatch):
    """The vectorized and pure-Python searches must return identical rankings."""

    if store_mod._np is None:
        pytest.skip("numpy not installed; only the python path is exercised")

    q = embedder.embed_one("how efficient are photovoltaic panels")

    store_np = VectorStore(":memory:")
    _populate(store_np, embedder)
    np_hits = store_np.search(q, top_k=4)

    # Force the pure-Python path on a fresh store.
    monkeypatch.setattr(store_mod, "_np", None)
    store_py = VectorStore(":memory:")
    _populate(store_py, embedder)
    py_hits = store_py.search(q, top_k=4)

    assert [h.source for h in np_hits] == [h.source for h in py_hits]
    for a, b in zip(np_hits, py_hits):
        assert a.score == pytest.approx(b.score, abs=1e-5)


def test_cache_invalidated_on_add_and_delete(embedder):
    store = VectorStore(":memory:")
    _populate(store, embedder)
    q = embedder.embed_one("espresso pressure")
    assert store.search(q, top_k=5)  # warms cache

    before = store.count_chunks()
    store.delete_source("coffee.md")
    hits = store.search(q, top_k=5)
    assert all(h.source != "coffee.md" for h in hits)
    assert store.count_chunks() < before

    # Adding a new doc must be visible immediately (cache rebuilt).
    chunks = chunk_text("A new note about espresso crema.", "new.md", chunk_size=200)
    store.add_chunks(chunks, embedder.embed([c.text for c in chunks]))
    hits2 = store.search(embedder.embed_one("crema"), top_k=5)
    assert any(h.source == "new.md" for h in hits2)
