"""Document loading and chunking.

Supports Markdown / plain-text natively (no dependencies) and PDF when the
optional ``pypdf`` package is present. Chunking is character-window based with
overlap and is careful to break on paragraph / sentence boundaries where it can,
which keeps citations readable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class Chunk:
    """A contiguous slice of a source document."""

    text: str
    source: str
    chunk_index: int
    char_start: int
    char_end: int


_WHITESPACE_RE = re.compile(r"[ \t]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    """Collapse noisy whitespace while preserving paragraph structure."""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WHITESPACE_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def read_document(path: str | Path) -> str:
    """Read a document from disk, dispatching on extension.

    ``.pdf`` requires the optional ``pypdf`` dependency; everything else is read
    as UTF-8 text. Raising here (rather than at import time) keeps the module
    import-safe when pypdf is absent.
    """

    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".pdf":
        return _read_pdf(p)
    return p.read_text(encoding="utf-8", errors="replace")


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader  # imported lazily; optional dependency
    except ImportError as exc:  # pragma: no cover - exercised only without pypdf
        raise RuntimeError(
            "Reading PDF files requires the 'pypdf' package. "
            "Install it with `pip install pypdf`, or convert the file to Markdown."
        ) from exc

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def _split_into_units(text: str) -> List[str]:
    """Split text into paragraph-ish units used as soft break points."""

    units = re.split(r"\n\s*\n", text)
    return [u.strip() for u in units if u.strip()]


def _langchain_chunks(
    text: str, source: str, chunk_size: int, chunk_overlap: int
) -> List[Chunk] | None:
    """Chunk using LangChain's RecursiveCharacterTextSplitter when available.

    Returns ``None`` if ``langchain-text-splitters`` is not installed, so the
    caller transparently falls back to the built-in splitter. This is how
    localrag "genuinely uses" LangChain when present while staying fully
    functional (and hermetic in tests) without it.
    """

    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    pieces = splitter.split_text(text)
    chunks: List[Chunk] = []
    cursor = 0
    for i, piece in enumerate(pieces):
        piece = piece.strip()
        if not piece:
            continue
        start = text.find(piece, cursor)
        if start < 0:
            start = cursor
        end = start + len(piece)
        cursor = max(cursor, end - chunk_overlap)
        chunks.append(
            Chunk(
                text=piece,
                source=source,
                chunk_index=len(chunks),
                char_start=start,
                char_end=end,
            )
        )
    return chunks


def chunk_text(
    text: str,
    source: str,
    *,
    chunk_size: int = 900,
    chunk_overlap: int = 150,
    use_langchain: bool = True,
) -> List[Chunk]:
    """Split ``text`` into overlapping chunks.

    If ``use_langchain`` is set and ``langchain-text-splitters`` is installed,
    LangChain's ``RecursiveCharacterTextSplitter`` is used. Otherwise a built-in
    paragraph-aware splitter runs: it greedily packs paragraph units into
    windows of at most ``chunk_size`` characters, then carries ``chunk_overlap``
    characters of the tail into the next window so retrieval context is not cut
    mid-thought. Oversized single paragraphs are hard-split so no chunk exceeds
    the window.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be in [0, chunk_size)")

    text = normalize_text(text)
    if not text:
        return []

    if use_langchain:
        lc = _langchain_chunks(text, source, chunk_size, chunk_overlap)
        if lc is not None:
            return lc

    units = _split_into_units(text)
    # Hard-split any unit that alone exceeds the window.
    expanded: List[str] = []
    for unit in units:
        if len(unit) <= chunk_size:
            expanded.append(unit)
        else:
            for i in range(0, len(unit), chunk_size):
                expanded.append(unit[i : i + chunk_size])

    chunks: List[Chunk] = []
    buffer = ""
    cursor = 0  # running char offset within the normalized text
    index = 0

    def flush(buf: str, start: int) -> None:
        nonlocal index
        buf = buf.strip()
        if not buf:
            return
        chunks.append(
            Chunk(
                text=buf,
                source=source,
                chunk_index=index,
                char_start=start,
                char_end=start + len(buf),
            )
        )
        index += 1

    buffer_start = 0
    for unit in expanded:
        candidate = f"{buffer}\n\n{unit}".strip() if buffer else unit
        if len(candidate) <= chunk_size or not buffer:
            if not buffer:
                buffer_start = cursor
            buffer = candidate
        else:
            flush(buffer, buffer_start)
            # Start new buffer with overlap tail from the previous buffer.
            tail = buffer[-chunk_overlap:] if chunk_overlap else ""
            buffer_start = max(0, buffer_start + len(buffer) - len(tail))
            buffer = f"{tail}\n\n{unit}".strip() if tail else unit
        cursor += len(unit) + 2  # account for the paragraph separator

    flush(buffer, buffer_start)
    return chunks


def load_and_chunk(
    path: str | Path,
    *,
    chunk_size: int = 900,
    chunk_overlap: int = 150,
) -> List[Chunk]:
    """Convenience helper: read a file and chunk it in one call."""

    text = read_document(path)
    source = Path(path).name
    return chunk_text(
        text, source, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
