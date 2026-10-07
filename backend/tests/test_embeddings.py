"""Tests for the hashing-vectorizer embedder and similarity."""

from __future__ import annotations

from app.embeddings import (
    HashingEmbedder,
    cosine_similarity,
    get_embedder,
    l2_normalize,
)


def test_embedding_is_deterministic():
    emb = HashingEmbedder(dim=128)
    a = emb.embed_one("the quick brown fox")
    b = emb.embed_one("the quick brown fox")
    assert a == b


def test_embedding_dimension():
    emb = HashingEmbedder(dim=64)
    v = emb.embed_one("some text here")
    assert len(v) == 64


def test_embedding_is_normalized():
    emb = HashingEmbedder(dim=128)
    v = emb.embed_one("normalization check with several tokens")
    norm = sum(x * x for x in v) ** 0.5
    assert abs(norm - 1.0) < 1e-6


def test_similar_texts_score_higher_than_unrelated():
    emb = HashingEmbedder(dim=512)
    solar_a = emb.embed_one("solar panels convert sunlight into electricity")
    solar_b = emb.embed_one("photovoltaic solar cells turn sunlight into power")
    coffee = emb.embed_one("espresso is brewed with nine bars of pressure")
    assert cosine_similarity(solar_a, solar_b) > cosine_similarity(solar_a, coffee)


def test_empty_text_yields_zero_vector():
    emb = HashingEmbedder(dim=32)
    v = emb.embed_one("")
    assert v == [0.0] * 32


def test_batch_embed_matches_single():
    emb = HashingEmbedder(dim=64)
    texts = ["alpha beta", "gamma delta"]
    batch = emb.embed(texts)
    singles = [emb.embed_one(t) for t in texts]
    assert batch == singles


def test_get_embedder_never_raises_offline():
    # prefer_model=False must always return the hashing fallback, no network.
    emb = get_embedder(prefer_model=False)
    assert emb.name == "hashing-vectorizer"


def test_l2_normalize_zero_vector():
    assert l2_normalize([0.0, 0.0]) == [0.0, 0.0]
