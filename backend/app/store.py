"""SQLite-backed document + vector store.

Vectors are stored as JSON arrays alongside their chunk text and provenance.
For a local, single-user Q&A tool this is simpler and more portable than a
dedicated vector DB, and the store is intentionally dependency-free (stdlib
``sqlite3`` + ``json``) so it works everywhere Python does.

**Search is vectorized when possible.** If ``numpy`` is installed, the store
loads every embedding into one ``float32`` matrix (cached across queries and
invalidated on write) and computes all cosine similarities with a single
matmul — orders of magnitude faster than a per-row Python loop as the corpus
grows. Without numpy it transparently falls back to a pure-Python cosine scan,
so the store still works with nothing but the standard library. See
``scripts/benchmark_search.py`` for latency-vs-corpus-size numbers.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from typing import List, Optional, Sequence

from .chunking import Chunk
from .embeddings import cosine_similarity

try:  # numpy is optional; the store degrades to a pure-Python scan without it.
    import numpy as _np
except Exception:  # pragma: no cover - exercised only when numpy is absent
    _np = None


@dataclass
class ScoredChunk:
    """A retrieved chunk plus its similarity score and DB id."""

    id: int
    text: str
    source: str
    chunk_index: int
    score: float


_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    source     TEXT NOT NULL,
    n_chunks   INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    source      TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    char_start  INTEGER NOT NULL,
    char_end    INTEGER NOT NULL,
    text        TEXT NOT NULL,
    embedding   TEXT NOT NULL,
    embed_dim   INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source);
"""


class VectorStore:
    """A thread-safe SQLite store for chunks and their embeddings."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        # check_same_thread=False + an explicit lock lets FastAPI's threadpool
        # share one connection safely.
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._lock = threading.Lock()
        # Cached in-memory matrix for vectorized search. Rebuilt lazily and
        # invalidated on every write.
        self._cache: Optional[dict] = None
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def _invalidate_cache(self) -> None:
        self._cache = None

    # ------------------------------------------------------------------ writes
    def add_chunks(
        self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]
    ) -> int:
        """Persist a document's chunks with their embeddings. Returns doc id."""

        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")
        if not chunks:
            raise ValueError("cannot add an empty document")

        source = chunks[0].source
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO documents (source, n_chunks) VALUES (?, ?)",
                (source, len(chunks)),
            )
            doc_id = int(cur.lastrowid)
            rows = [
                (
                    doc_id,
                    c.source,
                    c.chunk_index,
                    c.char_start,
                    c.char_end,
                    c.text,
                    json.dumps([round(float(x), 6) for x in emb]),
                    len(emb),
                )
                for c, emb in zip(chunks, embeddings)
            ]
            self._conn.executemany(
                """
                INSERT INTO chunks
                    (document_id, source, chunk_index, char_start, char_end,
                     text, embedding, embed_dim)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            self._conn.commit()
            self._invalidate_cache()
        return doc_id

    def delete_source(self, source: str) -> int:
        """Delete all documents/chunks for a given source filename."""

        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM documents WHERE source = ?", (source,)
            )
            self._conn.execute("DELETE FROM chunks WHERE source = ?", (source,))
            self._conn.commit()
            self._invalidate_cache()
            return cur.rowcount

    def clear(self) -> None:
        with self._lock:
            self._conn.executescript(
                "DELETE FROM chunks; DELETE FROM documents;"
            )
            self._conn.commit()
            self._invalidate_cache()

    # ------------------------------------------------------------------- reads
    def count_chunks(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()
        return int(row["n"])

    def list_sources(self) -> List[dict]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT source,
                       SUM(n_chunks) AS n_chunks,
                       MIN(created_at) AS created_at
                FROM documents
                GROUP BY source
                ORDER BY MIN(created_at) DESC
                """
            ).fetchall()
        return [dict(r) for r in rows]

    # ---------------------------------------------------------------- matrix
    def _load_rows(self) -> List[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(
                "SELECT id, source, chunk_index, text, embedding FROM chunks"
            ).fetchall()

    def _ensure_cache(self) -> Optional[dict]:
        """Build (or return) the cached numpy matrix of all embeddings.

        Returns ``None`` when numpy is unavailable, signalling the caller to use
        the pure-Python path. The cache holds the row metadata alongside the
        normalized ``float32`` matrix so a query is a single matmul.
        """

        if _np is None:
            return None
        if self._cache is not None:
            return self._cache
        rows = self._load_rows()
        if not rows:
            self._cache = {"rows": [], "matrix": None}
            return self._cache
        matrix = _np.asarray(
            [json.loads(r["embedding"]) for r in rows], dtype=_np.float32
        )
        # Normalize once so cosine reduces to a dot product.
        norms = _np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        matrix = matrix / norms
        self._cache = {"rows": rows, "matrix": matrix}
        return self._cache

    # ------------------------------------------------------------------ search
    def search(
        self,
        query_embedding: Sequence[float],
        top_k: int = 4,
        source: Optional[str] = None,
    ) -> List[ScoredChunk]:
        """Cosine search over stored chunks (vectorized when numpy is present).

        Optionally restrict to a single ``source``. Returns the ``top_k``
        highest-scoring chunks in descending score order.
        """

        if top_k <= 0:
            return []
        cache = self._ensure_cache()
        if cache is not None:
            return self._search_numpy(cache, query_embedding, top_k, source)
        return self._search_python(query_embedding, top_k, source)

    def _search_numpy(
        self,
        cache: dict,
        query_embedding: Sequence[float],
        top_k: int,
        source: Optional[str],
    ) -> List[ScoredChunk]:
        rows = cache["rows"]
        matrix = cache["matrix"]
        if matrix is None or not rows:
            return []
        q = _np.asarray(query_embedding, dtype=_np.float32)
        qn = _np.linalg.norm(q)
        if qn == 0.0:
            return []
        q = q / qn
        scores = matrix @ q  # (N,) cosine similarities in one matmul

        if source is not None:
            mask = _np.fromiter(
                (r["source"] == source for r in rows), dtype=bool, count=len(rows)
            )
            if not mask.any():
                return []
            scores = _np.where(mask, scores, -_np.inf)

        n = int((scores > -_np.inf).sum()) if source is not None else len(rows)
        k = min(top_k, n)
        if k <= 0:
            return []
        # argpartition for the top-k, then sort just those k.
        top_idx = _np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[_np.argsort(-scores[top_idx])]
        out: List[ScoredChunk] = []
        for i in top_idx:
            r = rows[int(i)]
            out.append(
                ScoredChunk(
                    id=int(r["id"]),
                    text=r["text"],
                    source=r["source"],
                    chunk_index=int(r["chunk_index"]),
                    score=float(scores[int(i)]),
                )
            )
        return out

    def _search_python(
        self,
        query_embedding: Sequence[float],
        top_k: int,
        source: Optional[str],
    ) -> List[ScoredChunk]:
        scored: List[ScoredChunk] = []
        for row in self._load_rows():
            if source is not None and row["source"] != source:
                continue
            emb = json.loads(row["embedding"])
            score = cosine_similarity(query_embedding, emb)
            scored.append(
                ScoredChunk(
                    id=int(row["id"]),
                    text=row["text"],
                    source=row["source"],
                    chunk_index=int(row["chunk_index"]),
                    score=float(score),
                )
            )
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:top_k]

    def close(self) -> None:
        with self._lock:
            self._conn.close()
