"""Seed the vector store with the repo's sample documents.

Used by the Docker image (and handy locally) so a fresh container comes up with
a populated corpus and the UI is immediately useful. Idempotent: it clears any
existing sample docs first so re-running doesn't duplicate chunks.

    python -m scripts.seed
    LOCALRAG_DB=/data/localrag.db python -m scripts.seed
"""

from __future__ import annotations

import sys
from pathlib import Path

from app.engine import RagEngine

# sample_docs/ lives at the repo root, two levels up from backend/app.
SAMPLE_DIR = Path(__file__).resolve().parents[2] / "sample_docs"


def main() -> int:
    if not SAMPLE_DIR.is_dir():
        print(f"! sample_docs not found at {SAMPLE_DIR}", file=sys.stderr)
        return 1

    engine = RagEngine()
    docs = sorted(SAMPLE_DIR.glob("*.md")) + sorted(SAMPLE_DIR.glob("*.txt"))
    if not docs:
        print(f"! no sample documents in {SAMPLE_DIR}", file=sys.stderr)
        return 1

    total = 0
    for path in docs:
        # Idempotent: drop any prior copy of this source before re-ingesting.
        engine.store.delete_source(path.name)
        result = engine.ingest_file(path)
        print(f"+ seeded {result.source}: {result.n_chunks} chunks")
        total += result.n_chunks

    print(f"done. {total} chunks indexed from {len(docs)} documents "
          f"({engine.embedder.name}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
