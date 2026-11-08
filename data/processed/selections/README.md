# Training-Set Selections

This directory contains the exact 15,000-image `train2017` manifests used by selection experiments.
The generated JSONL manifests and metadata sidecars are ignored by Git; this README is tracked so
the selection contract remains visible even when the local data artifacts are absent.

## Shared contract

Every implemented selection must:

- contain exactly 15,000 unique images from the canonical `train2017` manifest;
- use no `val2017` image, caption, statistic, or evaluation result during construction;
- record its seed, algorithm version, source-manifest SHA-256, and output SHA-256 in a metadata
  sidecar; and
- be generated before training rather than selecting images dynamically in the data loader.

## Method status

| Selection | Status | Local artifact |
|---|---|---|
| Random baseline | Implemented | `random_seed5942_n15000.jsonl` |
| Category balanced | Implemented | `category_balanced_t0p025_cap4_seed5942_n15000.jsonl` |
| Visual diversity | Placeholder — design pending | Not generated |
| Caption-semantic diversity | Placeholder — design pending | Not generated |
| Quality-aware diversity | Placeholder — design pending | Not generated |

## Random baseline

The random baseline uses seed `5942` and ranks every canonical training image by a stable keyed
hash. The first 15,000 ranked images form the baseline. Stable hashing makes the result independent
of source-row order and Python random-number-generator changes.

This same manifest is the control set for every selection-only comparison. It is also the shared
input for preprocessing-only experiments.

Generate it with:

```bash
make baseline
```

## Category-balanced selection

### Objective

The category-balanced method increases coverage of less-prevalent COCO object categories without
equalizing every category and without constructing an unrelated set from scratch. It begins with
the exact random baseline and deterministically swaps images only while an affected category target
is deficient.

For category `c`, the checked-in target is:

```text
frequency[c] = source_images_with_category[c] / 118,287

boost[c] = min(4, max(1, sqrt(0.025 / frequency[c])))

target[c] = min(
    source_images_with_category[c],
    ceil(15,000 * frequency[c] * boost[c])
)
```

The 2.5% threshold and 4× cap are project parameters calibrated from `train2017`; they are not
constants supplied by LVIS or another paper. At these settings, 36 categories receive boosted
targets. The selector scores an unselected image by the normalized deficits it would reduce across
all affected categories it contains, then removes a baseline image only when doing so preserves
targets already met. Seeded hashes resolve ties.

### Minimum-change requirement

The category-balanced treatment must replace at least 5% of the paired random baseline. “Changed”
means selected image IDs that are present in the category-balanced set but absent from the random
set:

```text
changed_count = size(category_balanced_ids - random_baseline_ids)
changed_fraction = changed_count / 15,000
required_changed_count = ceil(0.05 * 15,000) = 750
```

Because both sets have the same size, the same number of baseline images is removed. This measure
counts 948 replaced positions, not the 1,896 IDs in the two-sided symmetric difference.

The generator checks this requirement before writing either the JSONL or its metadata. If fewer
than 750 images change, generation fails. It does not add unrelated swaps merely to reach the
floor; the threshold must produce enough category-driven changes.

The current seed-5942 result is:

| Measurement | Result |
|---|---:|
| Images replaced relative to random | 948 / 15,000 (6.32%) |
| Images retained from random | 14,052 / 15,000 (93.68%) |
| Required minimum | 750 / 15,000 (5.00%) |
| Affected-category targets met | 36 / 36 |
| Mean distinct categories per image | 2.919 random → 3.021 balanced |

Generate and audit it with:

```bash
make analyze-category-balance
make category-balanced
make validate
```

The analysis report is written to `reports/data/category-balance-analysis.md`. The selection
metadata repeats the threshold, cap, minimum-change fraction, actual overlap, changed count, and
whether the requirement was met.

## Future selection placeholders

These methods remain intentionally unresolved. Research notes in `docs/background-research.md` are
starting points, not approved operational specifications. Each method should receive its own
parameter decision, implementation, real-data audit, metadata contract, and tests before its
placeholder is replaced.

### Visual diversity — TODO

- Decide and pin the visual encoder checkpoint and preprocessing revision.
- Define the diversity objective and scalable optimizer.
- Define duplicate handling, deterministic tie-breaking, and audit metrics.
- Do not generate a manifest yet.

### Caption-semantic diversity — TODO

- Decide and pin the text encoder and caption-aggregation rule.
- Define the semantic-diversity objective and scalable optimizer.
- Define deterministic tie-breaking and category-retention audits.
- Do not generate a manifest yet.

### Quality-aware diversity — TODO

- Decide the quality signals, rejection policy, and diversity–quality trade-off.
- Define score normalization, missing-value behavior, and deterministic optimization.
- Audit which categories and image conditions the quality rule removes.
- Do not generate a manifest yet.
