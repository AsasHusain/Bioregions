# World Atlas of Biotic Regions: method plan

This plan is fixed before the first map of the atlas is built. Changes to it are recorded in
[`ledger.md`](ledger.md), and the test rules are registered in [`preregistration.md`](preregistration.md) before any
result is seen.

## Definition

A Biotic Region is a connected area whose native plants and animals, taken together, form a community distinct from
those of its neighbours. Each region holds a characteristic mix of habitats. Size neither qualifies nor disqualifies
a region: a small, coherent area with a distinctive native biota is eligible.

The target is the native biota of the recent historical period, before modern land conversion:

- **Plants:** records of species native to the region only, checked against the native ranges of the World
  Checklist of Vascular Plants; planted and cultivated records are excluded.
- **Animals:** native ranges of mammals, birds, reptiles and amphibians, including ranges where a species is recently
  extinct (reported with and without them).
- **Uncertain places:** cells where most land is converted, such as irrigated plains and oases, are flagged. There,
  surviving records describe what remains, not the original community.

## Evidence

All evidence sits on one 10 km equal-area grid covering the whole mapped area in one run, so the map has no seams at
national borders.

| Evidence | Source |
|---|---|
| Plant composition | A dissimilarity model of native plant turnover, fitted on plant records and vegetation plots |
| Animal composition | Range maps, with birds, mammals, and reptiles with amphibians weighted equally |
| Vegetation mix | The share of each natural vegetation type around each cell, from the 90 m vegetation map |
| Environment | Climate, terrain, soils and geology |

The kinds of evidence overlap: the vegetation map and the plant model both draw on climate. They are therefore added
in steps, starting from plants and animals alone, and each addition is kept only if it improves predictions on places
not used to build the map.

Regions are grown on the grid's neighbour graph, so similarity is computed between neighbouring cells and to region
centres, never between every pair of cells.

## Evidence support

Each cell carries three measures:

- how far its conditions lie from those where plant data exist;
- how many records and plots lie nearby;
- how much the plant model's prediction varies when it is refitted.

Where plant support is weak, the map marks the cell provisional and says that its plant distinctions are modelled,
not observed.

## Testing

**Development.** Choices of method, weights, neighbourhood size, number of regions and boundary refinement are made
with spatial cross-validation. Each test fold's records are removed before the plant model is fitted and the map is
built. Choices are tuned inside the training folds only.

**Level of detail.** A region is split only where its parts differ in held-out species composition repeatedly across
folds, and the split loses no accuracy for any well-sampled group. Splits that cannot be tested for lack of data are
kept and marked provisional.

**Comparisons**, at equal numbers of regions:

| Comparison | Purpose |
|---|---|
| Regions from location alone | Null: nearness only |
| Regions from climate alone, and from climate and terrain | Nulls: environment only |
| The vegetation map and the animal map separately | The project's other designs |
| RESOLVE *Ecoregions 2017* | An established map, as an outside benchmark |

**Confirmation.** One chosen map is frozen, with its test plan, before sealed records are opened. They are opened
once. Before freezing, a script counts how many sealed test areas are usable, without revealing any species.

## Claims

Results are reported per group (plants, birds, mammals, reptiles) and support three separate claims:

1. **Useful:** the map captures biological structure that holds at new places, shown by beating the null comparisons.
2. **Best of the project's designs:** at least as good as each alternative for every well-sampled group.
3. **An improvement on established maps:** stated only if the map beats RESOLVE and other outside benchmarks.

If the map does not beat a null comparison with adequate precision, added value has not been shown under that test.
