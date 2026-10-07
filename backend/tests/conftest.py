"""Shared pytest fixtures.

All fixtures force ``prefer_model=False`` and an in-memory SQLite DB so the
suite is hermetic: no network, no model download, no on-disk state.
"""

from __future__ import annotations

import pytest

from app.engine import RagEngine
from app.embeddings import HashingEmbedder
from app.store import VectorStore

SAMPLE_DOCS = {
    "solar.md": (
        "# Solar Power\n\n"
        "Solar panels convert sunlight into electricity using photovoltaic cells. "
        "A typical residential rooftop installation ranges from 3 to 10 kilowatts.\n\n"
        "Photovoltaic efficiency for commercial silicon panels is usually between "
        "18 and 22 percent. Inverters convert the direct current produced by the "
        "panels into alternating current for the home.\n\n"
        "Battery storage systems such as lithium-ion packs let homeowners store "
        "excess energy generated during the day for use at night."
    ),
    "coffee.md": (
        "# Coffee Brewing\n\n"
        "Espresso is brewed by forcing hot water through finely ground coffee at "
        "roughly nine bars of pressure. The ideal water temperature is around 93 "
        "degrees Celsius.\n\n"
        "Pour-over methods like the V60 give a cleaner, brighter cup because the "
        "paper filter removes most oils. A common ratio is 15 grams of water per "
        "gram of coffee.\n\n"
        "Cold brew steeps coarse grounds in cold water for 12 to 24 hours, "
        "producing a low-acidity concentrate."
    ),
}


@pytest.fixture()
def engine() -> RagEngine:
    """A fresh in-memory engine using the deterministic hashing embedder."""

    eng = RagEngine(
        embedder=HashingEmbedder(dim=256),
        store=VectorStore(":memory:"),
    )
    return eng


@pytest.fixture()
def populated_engine(engine: RagEngine) -> RagEngine:
    for name, text in SAMPLE_DOCS.items():
        engine.ingest_text(text, source=name)
    return engine
