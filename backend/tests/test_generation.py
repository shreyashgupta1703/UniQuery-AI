"""Tests for the answer-generation subsystem.

Covers the parts that are hardest to get right and most user-facing: honest
citation selection/renumbering, the grounding signal, hybrid reranking, and the
streaming event protocol — all without a live Ollama server.
"""

from __future__ import annotations

from app import generation
from app.generation import (
    Answer,
    is_grounded,
    lexical_rerank,
    select_cited,
    stream_answer,
)
from app.store import ScoredChunk


def _chunk(cid, text, source, score):
    return ScoredChunk(id=cid, text=text, source=source, chunk_index=0, score=score)


# ----------------------------------------------------------- honest citations
def test_select_cited_keeps_only_referenced_and_renumbers():
    chunks = [
        _chunk(1, "solar panels are efficient", "solar.md", 0.9),
        _chunk(2, "coffee is brewed hot", "coffee.md", 0.5),
        _chunk(3, "water boils at 100C", "water.md", 0.4),
    ]
    text = "Panels are efficient [1]. Boiling matters [3]. Out of range [9]."
    rewritten, cites = select_cited(text, chunks)

    # Only [1] and [3] are valid; [3] is renumbered to [2]; [9] is dropped.
    assert [c.marker for c in cites] == [1, 2]
    assert [c.source for c in cites] == ["solar.md", "water.md"]
    assert "[1]" in rewritten and "[2]" in rewritten
    assert "[3]" not in rewritten and "[9]" not in rewritten


def test_select_cited_preserves_first_appearance_order():
    chunks = [_chunk(i, f"t{i}", f"s{i}.md", 0.5) for i in (1, 2, 3)]
    text = "First [3] then [1] then [3] again."
    rewritten, cites = select_cited(text, chunks)
    # [3] appears first -> becomes [1]; [1] appears second -> becomes [2].
    assert [c.source for c in cites] == ["s3.md", "s1.md"]
    assert rewritten == "First [1] then [2] then [1] again."


def test_select_cited_falls_back_to_all_when_no_markers():
    chunks = [_chunk(1, "a", "a.md", 0.9), _chunk(2, "b", "b.md", 0.8)]
    rewritten, cites = select_cited("No markers at all here.", chunks)
    assert len(cites) == 2
    assert rewritten == "No markers at all here."


# ------------------------------------------------------------------ grounding
def test_grounding_true_when_score_above_threshold():
    assert is_grounded([_chunk(1, "x", "a.md", generation.GROUNDING_THRESHOLD + 0.1)])


def test_grounding_false_when_all_scores_low():
    assert not is_grounded([_chunk(1, "x", "a.md", 0.01)])
    assert not is_grounded([])


# ------------------------------------------------------------------- reranking
def test_lexical_rerank_promotes_term_overlap():
    # Same vector score, but only one chunk literally contains the query terms.
    chunks = [
        _chunk(1, "unrelated filler text about nothing", "a.md", 0.30),
        _chunk(2, "espresso nine bars of pressure", "b.md", 0.30),
    ]
    out = lexical_rerank("espresso pressure", chunks, top_k=2)
    assert out[0].source == "b.md"


def test_lexical_rerank_respects_top_k():
    chunks = [_chunk(i, f"term{i}", f"s{i}.md", 0.5) for i in range(5)]
    assert len(lexical_rerank("term1", chunks, top_k=2)) == 2


# -------------------------------------------------------------------- extractive
def test_extractive_answer_is_grounded_flagged(populated_engine):
    ans = populated_engine.answer("How efficient are photovoltaic panels?", top_k=3)
    assert isinstance(ans, Answer)
    assert ans.mode == "extractive"
    assert ans.grounded is True
    assert ans.citations


# --------------------------------------------------------------------- streaming
def test_stream_answer_extractive_protocol():
    chunks = [_chunk(1, "Solar panels convert sunlight into power.", "solar.md", 0.9)]
    events = list(stream_answer("How do solar panels work?", chunks))

    types = [e.type for e in events]
    assert types[0] == "meta"
    assert "token" in types
    assert types[-1] == "citations"

    meta = events[0].data
    assert meta["mode"] == "extractive"
    assert meta["grounded"] is True

    text = "".join(e.data["text"] for e in events if e.type == "token")
    assert "Solar panels" in text or "solar" in text.lower()

    citations = events[-1].data["citations"]
    assert citations and citations[0]["source"] == "solar.md"


def test_stream_answer_empty_corpus_has_no_citations():
    events = list(stream_answer("anything", []))
    assert events[0].data["grounded"] is False
    assert events[-1].data["citations"] == []
    text = "".join(e.data["text"] for e in events if e.type == "token")
    assert "don't have" in text.lower()


# ----------------------------------------------------- ollama path (mocked)
class _FakeResp:
    """Minimal context-manager iterable mimicking Ollama's line-delimited JSON."""

    def __init__(self, lines):
        self._lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __iter__(self):
        return iter(self._lines)


def _ollama_lines(tokens, done_extra=None):
    import json

    out = [json.dumps({"response": t, "done": False}).encode() for t in tokens]
    out.append(json.dumps({"response": "", "done": True, **(done_extra or {})}).encode())
    return out


def test_generate_answer_ollama_honest_citations(monkeypatch):
    """The blocking Ollama path keeps only the chunks the model actually cited."""

    monkeypatch.setattr(generation.settings, "disable_ollama", False)
    monkeypatch.setattr(generation.settings, "ollama_model", "test-model")

    # Model answers citing only passage [2].
    tokens = ["Espresso ", "uses ", "nine ", "bars ", "[2]", "."]
    monkeypatch.setattr(
        generation.urllib.request, "urlopen", lambda *a, **k: _FakeResp(_ollama_lines(tokens))
    )

    chunks = [
        _chunk(1, "solar panels are efficient", "solar.md", 0.9),
        _chunk(2, "espresso is nine bars", "coffee.md", 0.8),
    ]
    ans = generation.generate_answer("How is espresso brewed?", chunks)
    assert ans.mode == "ollama"
    assert ans.model == "test-model"
    # Only the cited chunk survives, renumbered to [1].
    assert [c.source for c in ans.citations] == ["coffee.md"]
    assert "[1]" in ans.answer


def test_stream_answer_ollama_tokens(monkeypatch):
    """The streaming Ollama path forwards live tokens then resolves citations."""

    monkeypatch.setattr(generation.settings, "disable_ollama", False)
    monkeypatch.setattr(generation.settings, "ollama_model", "test-model")
    tokens = ["The ", "answer ", "[1]", "."]
    monkeypatch.setattr(
        generation.urllib.request, "urlopen", lambda *a, **k: _FakeResp(_ollama_lines(tokens))
    )

    chunks = [_chunk(1, "grounded passage text", "doc.md", 0.9)]
    events = list(generation.stream_answer("q?", chunks))
    assert events[0].data["mode"] == "ollama"
    text = "".join(e.data["text"] for e in events if e.type == "token")
    assert text == "The answer [1]."
    assert events[-1].data["citations"][0]["source"] == "doc.md"


def test_generate_answer_falls_back_on_ollama_error(monkeypatch):
    """Any transport failure cleanly degrades to the extractive path."""

    monkeypatch.setattr(generation.settings, "disable_ollama", False)

    def _boom(*a, **k):
        raise OSError("connection refused")

    monkeypatch.setattr(generation.urllib.request, "urlopen", _boom)
    chunks = [_chunk(1, "Solar panels convert sunlight.", "solar.md", 0.9)]
    ans = generation.generate_answer("How do panels work?", chunks)
    assert ans.mode == "extractive"
    assert ans.citations
