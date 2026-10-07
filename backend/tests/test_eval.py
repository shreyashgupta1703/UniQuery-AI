"""Gate CI on measured retrieval quality.

The eval harness computes Recall@k and MRR over a hand-labeled QA set. These
tests assert a minimum bar so a regression that quietly degrades retrieval fails
the build instead of shipping. They run fully offline with the hashing embedder.
"""

from __future__ import annotations

from app.embeddings import HashingEmbedder
from eval.harness import evaluate
from eval.qa import CASES


def test_hashing_embedder_meets_recall_bar():
    m = evaluate(HashingEmbedder(), k=4, rerank=True)
    assert m.n == len(CASES)
    # The hashing vectorizer must retrieve the right source for the vast
    # majority of questions within the top 4.
    assert m.recall_at_k >= 0.8, f"recall@4 regressed to {m.recall_at_k:.2f}"
    assert m.mrr >= 0.7, f"MRR regressed to {m.mrr:.3f}"


def test_rerank_does_not_hurt_recall():
    base = evaluate(HashingEmbedder(), k=4, rerank=False)
    reranked = evaluate(HashingEmbedder(), k=4, rerank=True)
    # Hybrid reranking should never drop below the pure-vector baseline.
    assert reranked.recall_at_k >= base.recall_at_k
    assert reranked.mrr >= base.mrr - 1e-9


def test_recall_at_1_is_reasonable():
    m = evaluate(HashingEmbedder(), k=1, rerank=True)
    # Even at k=1 the top result should usually be the right document.
    assert m.recall_at_k >= 0.6, f"recall@1 too low: {m.recall_at_k:.2f}"
