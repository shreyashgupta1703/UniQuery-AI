// Sample corpus bundled into the static demo so it works with zero backend.
// Mirrors sample_docs/ in the repo root.

export interface SampleDoc {
  name: string;
  text: string;
}

export const SAMPLE_DOCS: SampleDoc[] = [
  {
    name: "solar_power.md",
    text: `# Solar Power Basics

Solar panels convert sunlight into electricity using photovoltaic (PV) cells
made primarily of silicon. When photons strike the cell, they knock electrons
loose, creating a flow of direct current (DC).

## Efficiency

Commercial silicon panels typically achieve an efficiency of 18 to 22 percent,
meaning that fraction of the sunlight's energy is converted into usable
electricity. Premium monocrystalline panels sit at the top of that range, while
cheaper polycrystalline panels sit near the bottom.

## System Components

A grid-tied home system needs an inverter to convert the DC produced by the
panels into the alternating current (AC) used by household appliances and the
utility grid. Optional battery storage, usually lithium-ion, lets a household
store excess daytime energy for use after sunset or during outages.

## Sizing

A typical residential rooftop installation ranges from 3 to 10 kilowatts (kW).
As a rough rule of thumb, each kilowatt of panels requires about 5 to 7 square
meters of roof area and produces 3 to 5 kilowatt-hours per day depending on the
local climate and orientation.`,
  },
  {
    name: "coffee_brewing.md",
    text: `# Coffee Brewing Methods

Great coffee is mostly about grind size, water temperature, and the ratio of
coffee to water. Different methods emphasize different parts of the flavour.

## Espresso

Espresso is brewed by forcing hot water through a puck of finely ground coffee
at roughly nine bars of pressure. The ideal water temperature is around 93
degrees Celsius, and a typical shot uses about 18 grams of coffee to yield 36
grams of liquid in 25 to 30 seconds.

## Pour-Over

Pour-over methods such as the Hario V60 produce a cleaner, brighter cup because
the paper filter removes most of the coffee's oils and fine particles. A common
starting ratio is about 15 grams of water for every gram of coffee, poured in
slow, controlled stages.

## Cold Brew

Cold brew steeps coarse grounds in cold water for 12 to 24 hours. Because no
heat is used, far fewer acids and bitter compounds are extracted, producing a
smooth, low-acidity concentrate that is usually diluted with water or milk
before serving.`,
  },
];

export const SAMPLE_QUESTIONS = [
  "How efficient are commercial solar panels?",
  "What converts DC from solar panels into AC?",
  "At what pressure is espresso brewed?",
  "How long should cold brew steep?",
  "What is a good pour-over coffee ratio?",
];
