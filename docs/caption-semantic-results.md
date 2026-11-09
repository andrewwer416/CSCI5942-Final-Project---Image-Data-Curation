# Caption-semantic calibration results

The accepted selection uses frozen MPNet, the normalized mean of five normalized
caption vectors, exact GPU neighbors, sparse facility lazy greedy, and **k=50**.
It contains 15,000 unique train2017 images. **13,116 (87.44%)** replace images in
the seed-5942 random baseline; 1,884 overlap. All 591,435 captions were encoded
in FP32 without truncation. All four sweep candidates passed the hard gates.

| Metric | Random | Caption k=50 |
|---|---:|---:|
| Mean exact caption cosine coverage | 0.879811 | 0.902549 |
| Mean caption distance | 0.120189 | 0.097451 |
| p95 caption distance | 0.281780 | 0.209511 |
| p99 caption distance | 0.363018 | 0.247852 |

Mean caption distance improves about 18.9%; p95 improves about 25.6%.
These are encoder-space proxy improvements, not retrieval-model results.

| k | Mean exact coverage | p95 distance | p99 distance |
|---|---:|---:|---:|
| 50 | 0.902549 | 0.209511 | 0.247852 |
| 100 | 0.902782 | 0.208494 | 0.246452 |
| 200 | 0.902947 | 0.207913 | 0.245703 |
| 400 | 0.902993 | 0.207835 | 0.245480 |

k=50 meets every tolerance relative to the best candidate and leaves no source
image uncovered by its sparse graph. Larger k provides a small marginal gain.

## Category allocation

All **80/80 categories** have lower source-image mean and p95 caption coverage
distance than random. This does not mean every category requires more images.
For example:

| Category | Random count | Caption-selected count | Random mean distance | Selected mean distance |
|---|---:|---:|---:|---:|
| Giraffe | 327 | 155 | 0.070717 | 0.070473 |
| Tennis racket | 418 | 216 | 0.078261 | 0.075083 |
| Zebra | 242 | 129 | 0.076982 | 0.074351 |
| Broccoli | 276 | 148 | 0.090812 | 0.087041 |

Other labels receive more examples: tie 488 → 950, cell phone 584 → 945, bicycle
416 → 626, dog 588 → 850. Counts overlap because images have multiple labels.
This is evidence about coverage under the caption representation; it is not
causal evidence of model difficulty or the minimum samples needed to learn a label.

The complete table and interactive SVG tooltips are local artifacts:
`reports/data/caption-category-allocation.csv` and
`reports/data/caption-category-allocation.svg`.

## Language, redundancy, and agreement

| Descriptive proxy | Random | Caption-selected |
|---|---:|---:|
| Noun surface types | 7,727 | 7,967 |
| Verb surface types | 3,461 | 3,520 |
| Preposition-window types | 38,504 | 40,084 |
| Adjective surface types | 2,942 | 2,905 |
| Numeric-phrase types | 1,938 | 1,858 |
| Fixed scene-lexicon types | 22 | 22 |
| Repeated-source-caption occurrences | 4,135 | 2,972 |
| Mean selected nearest-other semantic similarity | 0.862385 | 0.810491 |
| Mean within-image caption dispersion | 0.371052 | 0.378444 |

The objective improves semantic coverage without improving every lexical proxy.
Adjective and numeric diversity decrease slightly. Preposition-window type counts
increase while covered source-incidence mass falls from 0.80490 to 0.80149.
These distinctions matter: more unique wording does not guarantee better coverage
of common language patterns. POS tags, surface forms, fixed scene words, and short
windows are noisy proxies, not gold semantic parses.

## Outliers and cross-view checks

The bottom 1% of source semantic density contains 1,183 images. The selected set
contains **1,098 (7.32% of selection)**, versus random's 153 (1.02%). Thus about
92.8% of the extreme source tail is selected. This is a material concentration,
not a dormant diagnostic. No outlier quota was used or retroactively introduced.

Review of the 50 lowest-density caption examples found coherent unusual content,
including sumo wrestling, spiders, a post office, bubble blowing, fire juggling,
glassmaking, and first aid. Some contain annotator disagreement or typos. This
review assesses caption content, not image-caption correctness, and does not prove
that every enriched outlier is useful. The full five-caption examples and selected
flags are preserved in the diagnostics JSON for further inspection.

In the independent visual CLIP space, mean distance changes from 0.200748 to
0.199671 and p95 from 0.358660 to 0.348998. Thus visual coverage improves modestly,
much less than the direct visual-selection method. Raw distances across the two
encoders are not directly comparable; compare changes within each encoder.

## Reproduction and provenance

See [the method contract](caption-semantic-selection.md). Run `make caption-semantic`
and `make caption-diagnostics` in the locked Pixi environment. The accepted local
manifest is `data/processed/selections/caption_semantic_k50_seed5942_n15000.jsonl`.
The selection passes canonical record, caption, hash, and split validation.

`reports/data/caption-calibration.json` records all candidate metrics and selection
provenance; `caption-k{k}.json` records every candidate's selected IDs.
`caption-diagnostics.json` records all phrase/category counts, outliers, dispersion,
redundancy, cross-view measurements, source/selection hashes, and tagger hashes.
Generated artifacts remain ignored by Git. The tracked code and lock file recreate them.
