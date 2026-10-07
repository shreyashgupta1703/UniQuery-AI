"""End-to-end retrieval + answer tests (network-free).

These are the core correctness tests for the RAG pipeline: ingest documents,
retrieve the right chunks for a query, and produce a cited extractive answer
without any LLM or network.
"""

from __future__ import annotations

import os

from app.generation import extractive_answer


def test_ingest_reports_chunks(engine):
    result = engine.ingest_text("Some short document about testing.", "t.md")
    assert result.n_chunks >= 1
    assert result.source == "t.md"
    assert engine.store.count_chunks() == result.n_chunks


def test_retrieval_returns_relevant_source(populated_engine):
    hits = populated_engine.retrieve("How efficient are photovoltaic panels?", top_k=3)
    assert hits, "expected at least one hit"
    # The top hit should come from the solar document, not coffee.
    assert hits[0].source == "solar.md"
    assert hits[0].score >= hits[-1].score  # sorted descending


def test_retrieval_ranks_coffee_query_to_coffee(populated_engine):
    hits = populated_engine.retrieve("What pressure is espresso brewed at?", top_k=3)
    assert hits[0].source == "coffee.md"


def test_source_filter_restricts_results(populated_engine):
    hits = populated_engine.retrieve("energy", top_k=5, source="coffee.md")
    assert hits
    assert all(h.source == "coffee.md" for h in hits)


def test_top_k_is_respected(populated_engine):
    hits = populated_engine.retrieve("solar battery storage", top_k=2)
    assert len(hits) <= 2


def test_answer_is_cited_and_extractive(populated_engine):
    # Force the extractive path regardless of any Ollama on the machine.
    os.environ["LOCALRAG_DISABLE_OLLAMA"] = "1"
    from app import config, generation

    config.settings.disable_ollama = True

    ans = populated_engine.answer("How do solar inverters work?", top_k=3)
    assert ans.mode == "extractive"
    assert ans.citations, "answer must carry citations"
    # Citation markers must be sequential starting at 1.
    markers = [c.marker for c in ans.citations]
    assert markers == list(range(1, len(markers) + 1))
    # The answer text should reference at least the first citation.
    assert "[1]" in ans.answer
    # The winning citation should be the solar doc.
    assert any(c.source == "solar.md" for c in ans.citations)


def test_extractive_answer_handles_no_chunks():
    ans = extractive_answer("anything", [])
    assert ans.mode == "extractive"
    assert ans.citations == []
    assert "don't have" in ans.answer.lower()


def test_delete_source_removes_chunks(populated_engine):
    before = populated_engine.store.count_chunks()
    removed = populated_engine.store.delete_source("coffee.md")
    assert removed >= 1
    after = populated_engine.store.count_chunks()
    assert after < before
    hits = populated_engine.retrieve("espresso", top_k=5)
    assert all(h.source != "coffee.md" for h in hits)


def test_stats_reports_embedder_and_sources(populated_engine):
    stats = populated_engine.stats()
    assert stats["embedder"] == "hashing-vectorizer"
    assert stats["n_chunks"] > 0
    sources = {s["source"] for s in stats["sources"]}
    assert {"solar.md", "coffee.md"} <= sources
