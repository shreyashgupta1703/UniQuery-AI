"""Tests for the built-in document chunker.

These pin ``use_langchain=False`` so they always exercise localrag's own
paragraph-aware splitter regardless of whether LangChain happens to be
installed. The optional LangChain path is covered separately in
``test_langchain_splitter.py``.
"""

from __future__ import annotations

import functools

import pytest

from app.chunking import Chunk, normalize_text
from app.chunking import chunk_text as _chunk_text

# Always test the built-in algorithm here.
chunk_text = functools.partial(_chunk_text, use_langchain=False)


def test_normalize_collapses_whitespace():
    raw = "hello    world\r\n\r\n\r\n\r\nfoo\tbar"
    out = normalize_text(raw)
    assert "    " not in out
    assert "\t" not in out
    assert "\n\n\n" not in out
    assert "hello world" in out


def test_chunking_respects_size():
    text = "\n\n".join(f"Paragraph number {i} with some filler words." for i in range(50))
    chunks = chunk_text(text, "doc.md", chunk_size=200, chunk_overlap=40)
    assert len(chunks) > 1
    for c in chunks:
        assert isinstance(c, Chunk)
        # Allow a small slack for the overlap tail + separator.
        assert len(c.text) <= 200 + 40 + 4


def test_chunk_indices_are_sequential():
    text = "\n\n".join(f"Block {i} " * 20 for i in range(20))
    chunks = chunk_text(text, "doc.md", chunk_size=300, chunk_overlap=50)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert all(c.source == "doc.md" for c in chunks)


def test_overlap_carries_context():
    text = "\n\n".join(f"Sentence group {i}. " * 10 for i in range(8))
    chunks = chunk_text(text, "doc.md", chunk_size=250, chunk_overlap=60)
    # Consecutive chunks should share some overlapping text.
    assert len(chunks) >= 2
    shared_any = False
    for a, b in zip(chunks, chunks[1:]):
        tail = a.text[-40:]
        if tail.split() and any(w in b.text for w in tail.split()[:3]):
            shared_any = True
    assert shared_any


def test_oversized_paragraph_is_hard_split():
    giant = "x" * 5000
    chunks = chunk_text(giant, "doc.md", chunk_size=500, chunk_overlap=0)
    assert len(chunks) >= 10
    assert all(len(c.text) <= 500 for c in chunks)


def test_empty_text_yields_no_chunks():
    assert chunk_text("   \n\n  ", "doc.md") == []


def test_invalid_overlap_raises():
    with pytest.raises(ValueError):
        chunk_text("hello", "doc.md", chunk_size=100, chunk_overlap=100)
