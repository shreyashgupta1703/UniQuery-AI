"""Tests for the optional LangChain splitter integration.

These skip cleanly when ``langchain-text-splitters`` is not installed, keeping
the core suite hermetic while still exercising the integration when present.
"""

from __future__ import annotations

import pytest

pytest.importorskip("langchain_text_splitters")

from app.chunking import chunk_text  # noqa: E402


def _make_text() -> str:
    return "\n\n".join(
        f"Paragraph {i}. " + ("word " * 40).strip() for i in range(30)
    )


def test_langchain_path_produces_bounded_chunks():
    text = _make_text()
    chunks = chunk_text(
        text, "doc.md", chunk_size=300, chunk_overlap=50, use_langchain=True
    )
    assert len(chunks) > 1
    # RecursiveCharacterTextSplitter respects the size budget.
    assert all(len(c.text) <= 300 + 5 for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert all(c.source == "doc.md" for c in chunks)


def test_builtin_and_langchain_both_cover_text():
    text = _make_text()
    lc = chunk_text(text, "d.md", chunk_size=300, chunk_overlap=50, use_langchain=True)
    builtin = chunk_text(
        text, "d.md", chunk_size=300, chunk_overlap=50, use_langchain=False
    )
    # Both paths should chunk the document into multiple pieces.
    assert len(lc) > 1
    assert len(builtin) > 1
    # A distinctive token from the text must survive in both chunkings.
    assert any("Paragraph 0" in c.text for c in lc)
    assert any("Paragraph 0" in c.text for c in builtin)
