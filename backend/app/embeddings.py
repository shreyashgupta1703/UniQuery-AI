"""Local text embeddings with a zero-dependency fallback.

Primary path: ``sentence-transformers`` (a Hugging Face model) produces dense
semantic vectors. When the library or the model weights are unavailable — e.g.
an offline CI runner, a locked-down machine, or Python 3.14 before wheels
exist — we transparently fall back to a deterministic **hashing vectorizer**.

The hashing vectorizer is a real, useful embedding: it maps token n-grams into a
fixed-dimensional space via feature hashing with signed buckets, then L2
normalizes. It requires no model download, no network, and no numpy — pure
standard library — so retrieval always works and tests are hermetic.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import List, Sequence

from .config import settings

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def _ngrams(tokens: Sequence[str], n: int) -> List[str]:
    if n <= 1:
        return list(tokens)
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def l2_normalize(vec: List[float]) -> List[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        return vec
    return [v / norm for v in vec]


class HashingEmbedder:
    """Deterministic feature-hashing embedder (no external deps).

    Uses unigrams + bigrams, signed feature hashing (the "hashing trick") into
    ``dim`` buckets, and L2 normalization so cosine similarity reduces to a dot
    product. Identical text always yields an identical vector, which makes the
    retrieval tests fully reproducible and network-free.
    """

    name = "hashing-vectorizer"

    def __init__(self, dim: int | None = None) -> None:
        self.dim = int(dim or settings.hash_dim)
        if self.dim <= 0:
            raise ValueError("embedding dim must be positive")

    def _hash(self, feature: str) -> tuple[int, float]:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        h = int.from_bytes(digest, "big")
        bucket = h % self.dim
        sign = 1.0 if (h >> 63) & 1 else -1.0
        return bucket, sign

    def embed_one(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        tokens = _tokenize(text)
        if not tokens:
            return vec
        features = _ngrams(tokens, 1) + _ngrams(tokens, 2)
        for feat in features:
            bucket, sign = self._hash(feat)
            vec[bucket] += sign
        return l2_normalize(vec)

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        return [self.embed_one(t) for t in texts]


class SentenceTransformerEmbedder:
    """Wraps a Hugging Face sentence-transformers model when available."""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer  # lazy import

        self._model = SentenceTransformer(model_name)
        self.name = f"sentence-transformers::{model_name}"
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        vectors = self._model.encode(
            list(texts), normalize_embeddings=True, convert_to_numpy=False
        )
        return [[float(x) for x in row] for row in vectors]

    def embed_one(self, text: str) -> List[float]:
        return self.embed([text])[0]


def get_embedder(prefer_model: bool = True):
    """Return the best available embedder.

    Tries sentence-transformers first (unless ``prefer_model`` is False), and
    falls back to the hashing vectorizer on any import/load failure. This
    function never raises for lack of the optional model.
    """

    if prefer_model:
        try:
            return SentenceTransformerEmbedder(settings.embed_model)
        except Exception:  # noqa: BLE001 - any failure => graceful fallback
            pass
    return HashingEmbedder()


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity for two vectors (assumed roughly normalized)."""

    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)
