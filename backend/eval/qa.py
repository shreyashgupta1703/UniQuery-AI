"""Hand-labeled QA set for retrieval evaluation.

Each case pairs a natural-language question with the source document that
contains the answer, over the two docs in ``sample_docs/``. Small but honest:
these are the questions a user would actually ask, and the labels are the
ground-truth source the answer must be retrieved from.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class QACase:
    question: str
    expected_source: str


# The evaluated corpus (kept inline so the harness is self-contained and the
# labels can't drift from the text they were written against).
CORPUS: dict[str, str] = {
    "solar.md": (
        "# Solar Power Basics\n\n"
        "Solar panels convert sunlight into electricity using photovoltaic (PV) "
        "cells made primarily of silicon. When photons strike the cell, they "
        "knock electrons loose, creating a flow of direct current (DC).\n\n"
        "## Efficiency\n\n"
        "Commercial silicon panels typically achieve an efficiency of 18 to 22 "
        "percent, meaning that fraction of the sunlight's energy is converted "
        "into usable electricity. Premium monocrystalline panels sit at the top "
        "of that range, while cheaper polycrystalline panels sit near the bottom."
        "\n\n## System Components\n\n"
        "A grid-tied home system needs an inverter to convert the DC produced by "
        "the panels into the alternating current (AC) used by household "
        "appliances and the utility grid. Optional battery storage, usually "
        "lithium-ion, lets a household store excess daytime energy for use after "
        "sunset or during outages.\n\n"
        "## Sizing\n\n"
        "A typical residential rooftop installation ranges from 3 to 10 "
        "kilowatts (kW). As a rough rule of thumb, each kilowatt of panels "
        "requires about 5 to 7 square meters of roof area and produces 3 to 5 "
        "kilowatt-hours per day depending on the local climate and orientation."
    ),
    "coffee.md": (
        "# Coffee Brewing Methods\n\n"
        "Great coffee is mostly about grind size, water temperature, and the "
        "ratio of coffee to water. Different methods emphasize different parts "
        "of the flavour.\n\n"
        "## Espresso\n\n"
        "Espresso is brewed by forcing hot water through a puck of finely ground "
        "coffee at roughly nine bars of pressure. The ideal water temperature is "
        "around 93 degrees Celsius, and a typical shot uses about 18 grams of "
        "coffee to yield 36 grams of liquid in 25 to 30 seconds.\n\n"
        "## Pour-Over\n\n"
        "Pour-over methods such as the Hario V60 produce a cleaner, brighter cup "
        "because the paper filter removes most of the coffee's oils and fine "
        "particles. A common starting ratio is about 15 grams of water for every "
        "gram of coffee, poured in slow, controlled stages.\n\n"
        "## Cold Brew\n\n"
        "Cold brew steeps coarse grounds in cold water for 12 to 24 hours. "
        "Because no heat is used, far fewer acids and bitter compounds are "
        "extracted, producing a smooth, low-acidity concentrate that is usually "
        "diluted with water or milk before serving."
    ),
}


CASES: List[QACase] = [
    QACase("How efficient are commercial silicon solar panels?", "solar.md"),
    QACase("What converts the DC from solar panels into AC?", "solar.md"),
    QACase("How much roof area does each kilowatt of panels need?", "solar.md"),
    QACase("What stores excess solar energy for use after sunset?", "solar.md"),
    QACase("How do photovoltaic cells generate electricity?", "solar.md"),
    QACase("At what pressure is espresso brewed?", "coffee.md"),
    QACase("What water temperature is ideal for espresso?", "coffee.md"),
    QACase("Why does pour-over coffee taste cleaner?", "coffee.md"),
    QACase("How long should cold brew steep?", "coffee.md"),
    QACase("What is a good coffee-to-water ratio for pour-over?", "coffee.md"),
]
