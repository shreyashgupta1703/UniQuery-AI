"""The RAG engine: orchestrates ingest -> embed -> store -> retrieve -> answer.

This is the single object the API (and tests) talk to. It owns one embedder and
one vector store, and exposes high-level operations. Everything here works fully
offline; the only optional online-ish piece is Ollama generation, which degrades
to the extractive answerer automatically.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .chunking import chunk_text, load_and_chunk
from .config import Settings, settings as default_settings
from .embeddings import get_embedder
from .generation import Answer, generate_answer, lexical_rerank, stream_answer
from .store import ScoredChunk, VectorStore


@dataclass
class IngestResult:
    source: str
    n_chunks: int
    document_id: int


class RagEngine:
    """High-level retrieval-augmented generation engine."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        *,
        prefer_model: bool = True,
        embedder=None,
        store: Optional[VectorStore] = None,
    ) -> None:
        self.settings = settings or default_settings
        # ``prefer_model=False`` (or an injected embedder) keeps tests hermetic.
        self.embedder = embedder or get_embedder(prefer_model=prefer_model)
        self.store = store or VectorStore(self.settings.db_path)

    # --------------------------------------------------------------- ingestion
    def ingest_text(self, text: str, source: str) -> IngestResult:
        """Chunk, embed and store a raw text document."""

        chunks = chunk_text(
            text,
            source,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        if not chunks:
            raise ValueError("document produced no chunks (is it empty?)")
        embeddings = self.embedder.embed([c.text for c in chunks])
        doc_id = self.store.add_chunks(chunks, embeddings)
        return IngestResult(source=source, n_chunks=len(chunks), document_id=doc_id)

    def ingest_file(self, path: str | Path) -> IngestResult:
        """Read, chunk, embed and store a file (Markdown/text/PDF)."""

        chunks = load_and_chunk(
            path,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        if not chunks:
            raise ValueError(f"file produced no chunks: {path}")
        embeddings = self.embedder.embed([c.text for c in chunks])
        doc_id = self.store.add_chunks(chunks, embeddings)
        return IngestResult(
            source=Path(path).name, n_chunks=len(chunks), document_id=doc_id
        )

    # --------------------------------------------------------------- retrieval
    def retrieve(
        self,
        question: str,
        top_k: Optional[int] = None,
        source: Optional[str] = None,
        *,
        rerank: bool = True,
    ) -> List[ScoredChunk]:
        """Embed the question and return the top matching chunks.

        When ``rerank`` is set, the store is over-fetched (``top_k * 3``
        candidates) and a cheap lexical/vector hybrid pass re-sorts them before
        the final ``top_k`` are returned. This measurably improves precision on
        keyword-heavy questions (see ``eval/``) at negligible cost.
        """

        k = top_k if top_k is not None else self.settings.top_k
        query_vec = self.embedder.embed_one(question)
        if not rerank:
            return self.store.search(query_vec, top_k=k, source=source)
        candidates = self.store.search(query_vec, top_k=k * 3, source=source)
        return lexical_rerank(question, candidates, k)

    # ------------------------------------------------------------------ answer
    def answer(
        self, question: str, top_k: Optional[int] = None, source: Optional[str] = None
    ) -> Answer:
        """Full pipeline: retrieve context then generate a cited answer."""

        chunks = self.retrieve(question, top_k=top_k, source=source)
        return generate_answer(question, chunks)

    def answer_stream(
        self, question: str, top_k: Optional[int] = None, source: Optional[str] = None
    ):
        """Full pipeline, streamed: retrieve then yield answer events live."""

        chunks = self.retrieve(question, top_k=top_k, source=source)
        return stream_answer(question, chunks)

    # ------------------------------------------------------------------- admin
    def stats(self) -> dict:
        return {
            "embedder": self.embedder.name,
            "embed_dim": getattr(self.embedder, "dim", None),
            "n_chunks": self.store.count_chunks(),
            "sources": self.store.list_sources(),
        }
