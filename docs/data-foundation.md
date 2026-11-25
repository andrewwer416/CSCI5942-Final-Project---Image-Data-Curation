# Data Foundation

## Scope

This document is the data contract: auditable inputs, manifests, selections and the leakage boundary.
The model, image transforms and pretrained embeddings are documented separately (see
`docs/study-methods.md`). Tests never download large archives.
Official COCO archive links use HTTP; acquisition therefore verifies pinned byte counts and SHA-256
digests before safe extraction.

## Reproducible flow

```text
official COCO archives
        ↓ immutable extraction
captions + instances + image metadata
        ↓ deterministic join
canonical train2017 / val2017 manifests
        ↓ versioned selection method + seed
exact 15,000-image training manifests
        ↓ run manifest + lazy image transform
training and retrieval evaluation
```

The canonical layer separates data discovery from experiment selection. A run should reference a
selection-manifest hash rather than selecting images inside the training process.

## Canonical manifest contract

Each JSONL row represents one image and has these fields:

| Field | Type | Meaning |
|---|---|---|
| `image_id` | integer | Official COCO image identifier |
| `split` | string | `train2017` or `val2017` |
| `file_name` | string | Official image filename |
| `relative_path` | string | Path relative to `data/raw/coco/` |
| `width`, `height` | integer | Original pixel dimensions |
| `captions` | array | Five `{caption_id, text}` records |
| `category_ids` | integer array | Sorted distinct COCO object categories present |
| `object_count` | integer | Number of instance annotations for the image |

Rows are sorted by image ID and serialized with stable JSON key ordering. The metadata sidecar
records input checksums, output checksum, row counts, caption distribution, and the COCO category
dictionary. A small number of official records can contain more than five captions; to preserve the
study's exact-five contract, preparation retains the five lowest caption IDs and records both the raw
count distribution and this policy in metadata. Any image with fewer than five captions fails early.

## Selection manifests

Selection outputs use the same row contract, so training can consume either canonical or selected
manifests with one reader. Their metadata additionally records:

- selection method and algorithm version;
- seed and requested budget;
- canonical-manifest SHA-256;
- selected-manifest SHA-256; and
- per-category image coverage.

The random baseline ranks each image by a stable keyed hash. It does not depend on source row order
or Python's evolving pseudo-random sampling implementation.

The category-balanced manifest is paired with that exact random baseline. At least 5% of its 15,000
selected IDs must be replacements, so at least 750 category-balanced IDs must be absent from the
random set. This is a hard generation requirement: the command fails before writing a manifest if
the target-driven swaps do not clear the floor, and it never pads the result with unrelated swaps.
The current calibrated selection replaces 948 images (6.32%). The complete operational definitions
of all five selections live in `data/processed/selections/README.md`.

## Caption sampling

Training chooses one of an image's five captions from a hash of `(seed, epoch, image_id)`. The
choice is reproducible, independent of data-loader worker order, and does not mutate global random
state. Evaluation retains all five captions.

## Validation and leakage boundary

`vlm-data validate` checks schema-level integrity, exact caption counts, manifest hashes, expected
split sizes, selection size, uniqueness, membership in the canonical training source, and train/val
ID disjointness. `--check-files` adds an on-disk existence check after image archives are installed.

No artifact derived from `val2017` may influence training subset selection, normalization statistics,
or early stopping. The proposal uses validation performance to choose optional follow-up pairings;
that makes those follow-ups adaptive to `val2017`. The final report should call this out, or reserve
an additional untouched evaluation split before running combinations.

## Method decisions (resolved)

The proposal intentionally left several method definitions open. All are now approved and pinned.
`docs/study-methods.md` gives the overview.

| Open question | Resolution | Record |
|---|---|---|
| Frozen CLIP checkpoint, preprocessing, precision, revision | OpenCLIP ViT-B-16 `datacomp_xl_s13b_b90k`, revision `767e6d14…`, native 224-px transform, AMP FP16 inference, normalized float32 cache | `docs/visual-calibration-results.md` |
| Visual-diversity objective and algorithm | Sparse facility location, exact GPU neighbors, lazy greedy, k = 50, duplicates at 0.99 | `docs/visual-calibration-results.md` |
| Caption embedding and aggregation | Frozen MPNet, mean of five normalized caption vectors, k = 50 | `docs/caption-semantic-selection.md` |
| Quality metrics and quality/diversity weighting | TOPIQ-NR filter (worst 0.25%, short side ≥ 192 px) plus a modular TOPIQ bonus, λ = 0.1099 | `docs/quality-aware-selection.md` |
| Crop ranges, interpolation, padding, color normalization | P3 image-relative crop (×0.8–1.25, 50–100%); PIL bilinear everywhere; P2 mid-gray padding with masking; P4 per-image Shades-of-Gray plus a brightness stretch | `docs/preprocessing-p2.md` to `docs/preprocessing-p4.md` |
| CLIP-vector fusion | Late fusion `projection(LayerNorm(CLS + W·clip))`, no gate, no dropout | `docs/preprocessing-p5.md` |
| Seed policy | Selection, caption, data-order and crop seeds fixed at 5942. Model seeds 5942–5944 for baseline replicates | `STATUS.md` |

See `background-research.md` for the literature review, concrete starting values, and train-only
calibration checks proposed for these decisions.

The category-balance decision is now resolved separately: `vlm-data analyze-category-balance`
evaluates its parameter grid from canonical `train2017`, verifies the real multi-label selection,
and writes SHA-pinned reports under `reports/data/`. The checked-in values are a 2.5% prevalence
threshold and an empirically inactive 4× safety cap, selected under a 14%-of-budget target-mass
guardrail and verified against a 5% minimum-change requirement.
