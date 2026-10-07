"""Compute and report retrieval-quality metrics.

Metrics (all @k, where a "hit" means the expected source appears among the
top-k retrieved chunks):

* **Hit-rate@k**  — fraction of questions with at least one correct-source hit.
* **Recall@k**    — same as hit-rate here (one relevant source per question).
* **MRR**         — mean reciprocal rank of the first correct-source hit.

Run it as a report:

    python -m eval.harness
    python -m eval.harness --k 3 --rerank

The same functions are imported by ``tests/test_eval.py`` to gate CI on a
minimum score, so the quality claim is enforced, not just asserted.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Optional

from app.embeddings import HashingEmbedder
from app.engine import RagEngine
from app.store import VectorStore

from .qa import CASES, CORPUS, QACase


@dataclass
class Metrics:
    embedder: str
    k: int
    rerank: bool
    n: int
    hit_rate: float
    recall_at_k: float
    mrr: float


def _build_engine(embedder) -> RagEngine:
    eng = RagEngine(embedder=embedder, store=VectorStore(":memory:"))
    for name, text in CORPUS.items():
        eng.ingest_text(text, source=name)
    return eng


def _first_correct_rank(hits, expected: str) -> Optional[int]:
    for rank, h in enumerate(hits, start=1):
        if h.source == expected:
            return rank
    return None


def evaluate(
    embedder,
    *,
    k: int = 4,
    rerank: bool = True,
    cases: List[QACase] = CASES,
) -> Metrics:
    """Run every QA case through retrieval and aggregate the metrics."""

    eng = _build_engine(embedder)
    hits_count = 0
    reciprocal_sum = 0.0
    for case in cases:
        hits = eng.retrieve(case.question, top_k=k, rerank=rerank)
        rank = _first_correct_rank(hits, case.expected_source)
        if rank is not None:
            hits_count += 1
            reciprocal_sum += 1.0 / rank
    n = len(cases)
    return Metrics(
        embedder=embedder.name,
        k=k,
        rerank=rerank,
        n=n,
        hit_rate=hits_count / n,
        recall_at_k=hits_count / n,
        mrr=reciprocal_sum / n,
    )


def _get_embedders():
    """Hashing embedder always; sentence-transformers if it loads."""

    embedders = [HashingEmbedder()]
    try:
        from app.embeddings import SentenceTransformerEmbedder
        from app.config import settings

        embedders.append(SentenceTransformerEmbedder(settings.embed_model))
    except Exception:  # noqa: BLE001 - optional model not installed
        pass
    return embedders


def format_table(rows: List[Metrics]) -> str:
    header = f"{'embedder':<34} {'k':>2} {'rerank':>7} {'hit':>6} {'recall':>7} {'MRR':>6}"
    lines = [header, "-" * len(header)]
    for m in rows:
        lines.append(
            f"{m.embedder:<34} {m.k:>2} {str(m.rerank):>7} "
            f"{m.hit_rate:>6.2f} {m.recall_at_k:>7.2f} {m.mrr:>6.3f}"
        )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate localrag retrieval.")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument(
        "--rerank",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="force rerank on/off (default: report both)",
    )
    args = ap.parse_args()

    rerank_modes = [args.rerank] if args.rerank is not None else [False, True]
    rows: List[Metrics] = []
    for embedder in _get_embedders():
        for rerank in rerank_modes:
            rows.append(evaluate(embedder, k=args.k, rerank=rerank))

    print(f"\nlocalrag retrieval eval — {len(CASES)} labeled questions "
          f"over {len(CORPUS)} docs\n")
    print(format_table(rows))
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
