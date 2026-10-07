"""Benchmark vector-store search latency vs. corpus size.

Generates synthetic chunks, indexes them, and times ``VectorStore.search`` at a
range of corpus sizes for both the vectorized (numpy) and pure-Python paths so
you can see the cross-over. Run:

    python -m scripts.benchmark_search
    python -m scripts.benchmark_search --sizes 500 2000 10000 --queries 50

This proves the perf claim in the README ("orders of magnitude faster with
numpy") with real numbers on your own machine, and is safe to run anywhere —
it uses the dependency-free hashing embedder and an in-memory DB.
"""

from __future__ import annotations

import argparse
import random
import time
from statistics import mean

from app.chunking import Chunk
from app.embeddings import HashingEmbedder
from app import store as store_mod
from app.store import VectorStore

_WORDS = (
    "solar photovoltaic inverter battery grid coffee espresso brew filter roast "
    "water pressure temperature energy silicon panel storage lithium current "
    "voltage cell module renewable efficiency thermal capacity charge discharge"
).split()


def _random_text(rng: random.Random, n_words: int = 60) -> str:
    return " ".join(rng.choice(_WORDS) for _ in range(n_words))


def _build_store(n: int, embedder: HashingEmbedder, rng: random.Random) -> VectorStore:
    store = VectorStore(":memory:")
    batch = 200
    for start in range(0, n, batch):
        count = min(batch, n - start)
        chunks = [
            Chunk(
                text=_random_text(rng),
                source=f"doc{(start + i) % 25}.md",
                chunk_index=i,
                char_start=0,
                char_end=0,
            )
            for i in range(count)
        ]
        embs = embedder.embed([c.text for c in chunks])
        store.add_chunks(chunks, embs)
    return store


def _time_search(store: VectorStore, queries, top_k: int) -> float:
    # warm the cache
    store.search(queries[0], top_k=top_k)
    t0 = time.perf_counter()
    for q in queries:
        store.search(q, top_k=top_k)
    return (time.perf_counter() - t0) / len(queries) * 1000.0  # ms/query


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark localrag vector search.")
    ap.add_argument("--sizes", type=int, nargs="+", default=[200, 1000, 5000, 20000])
    ap.add_argument("--queries", type=int, default=25)
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    embedder = HashingEmbedder(dim=args.dim)
    queries = [embedder.embed_one(_random_text(rng)) for _ in range(args.queries)]

    has_numpy = store_mod._np is not None
    print(f"numpy available: {has_numpy}  (dim={args.dim}, top_k={args.top_k}, "
          f"{args.queries} queries/size)\n")
    print(f"{'corpus':>10} | {'numpy ms':>10} | {'python ms':>10} | {'speedup':>8}")
    print("-" * 48)

    for n in args.sizes:
        np_ms = py_ms = float("nan")

        if has_numpy:
            store_mod._np = _orig_np
            s = _build_store(n, embedder, rng)
            np_ms = _time_search(s, queries, args.top_k)
            s.close()

        # Force the pure-Python path.
        store_mod._np = None
        s = _build_store(n, embedder, rng)
        py_ms = _time_search(s, queries, args.top_k)
        s.close()
        store_mod._np = _orig_np  # restore

        speedup = (py_ms / np_ms) if (has_numpy and np_ms) else float("nan")
        print(f"{n:>10} | {np_ms:>10.3f} | {py_ms:>10.3f} | "
              f"{speedup:>7.1f}x" if has_numpy else
              f"{n:>10} | {'n/a':>10} | {py_ms:>10.3f} | {'n/a':>8}")

    return 0


_orig_np = store_mod._np

if __name__ == "__main__":
    raise SystemExit(main())
